#include "app/frame.hxx"

#include "app/app.hxx"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtx/string_cast.hpp>

namespace {
    auto max_in_flight_retire_value(AppResources const &res) -> u64 {
        u64 v = 0;
        for (auto const &fs: res.frames) {
            v = std::max(v, fs.frame_done_value);
        }
        return v;
    }
} // namespace

auto fill_frame_ubo_from_camera(FrameUBO &ubo, const EditorCamera &cam, VkExtent2D extent, float fov_y_radians,
                                float near_plane, float far_plane) -> void {
    const float aspect = static_cast<float>(extent.width) / std::max(1.0f, static_cast<float>(extent.height));

    ubo.view = cam.view_matrix();
    ubo.projection = glm::perspectiveRH_ZO(fov_y_radians, aspect, far_plane, near_plane);
    ubo.inv_projection = glm::inverse(ubo.projection);
    ubo.view_projection = ubo.projection * ubo.view;
    ubo.camera_position = glm::vec4(cam.camera_position(), 1.0f);
    ubo.inv_view_projection = glm::inverse(ubo.view_projection);
    ubo.inv_view_projection_no_translation = glm::inverse(ubo.projection * glm::mat4(glm::mat3(ubo.view)));
    ubo.viewport_size = glm::vec2(static_cast<float>(extent.width), static_cast<float>(extent.height));

    const auto planes = extract_frustum_planes(ubo.projection);

    ubo.frustum_planes = planes;
    std::swap(ubo.frustum_planes[4], ubo.frustum_planes[5]);
}

auto write_camera_to_frame_ubo(FrameUBO &ubo, RenderContext &ctx, AlignedRingBuffer<FrameUBO> &frame_ubo_ring,
                               u32 frame_index, const EditorCamera &cam, VkExtent2D extent, float fov_y_radians,
                               float near_plane, float far_plane) -> void {
    fill_frame_ubo_from_camera(ubo, cam, extent, fov_y_radians, near_plane, far_plane);
    frame_ubo_ring.write_element(ctx, frame_index, 0, ubo);
}
auto clamp_u32(u32 v, u32 lo, u32 hi) -> u32 { return std::min(std::max(v, lo), hi); }
auto sanitize_window_extent(VkExtent2D raw, VkPhysicalDevice physical_device, VkSurfaceKHR surface, ExtentBounds bounds)
        -> VkExtent2D {
    // If minimized, let caller decide to skip frame.
    if (raw.width == 0 || raw.height == 0) {
        return raw;
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical_device, &props);

    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &caps);

    const u32 dev_max = props.limits.maxImageDimension2D;

    const u32 max_w = std::min({bounds.max_dim, dev_max, caps.maxImageExtent.width});
    const u32 max_h = std::min({bounds.max_dim, dev_max, caps.maxImageExtent.height});

    const u32 min_w = std::max(bounds.min_dim, caps.minImageExtent.width);
    const u32 min_h = std::max(bounds.min_dim, caps.minImageExtent.height);

    raw.width = clamp_u32(raw.width, min_w, max_w);
    raw.height = clamp_u32(raw.height, min_h, max_h);
    return raw;
}
auto sanitize_scene_extent(VkExtent2D raw, VkExtent2D fallback_last_valid, VkPhysicalDevice physical_device,
                           ExtentBounds bounds) -> VkExtent2D {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical_device, &props);
    const u32 dev_max = props.limits.maxImageDimension2D;

    const u32 max_w = std::min(bounds.max_dim, dev_max);
    const u32 max_h = std::min(bounds.max_dim, dev_max);

    const u32 min_w = bounds.min_dim;
    const u32 min_h = bounds.min_dim;

    if (raw.width == 0 || raw.height == 0) {
        raw = fallback_last_valid;
    }

    raw.width = clamp_u32(raw.width, min_w, max_w);
    raw.height = clamp_u32(raw.height, min_h, max_h);
    return raw;
}
auto update_frame_timing(AppUI &ui) -> void {
    auto now = std::chrono::high_resolution_clock::now();
    ui.dt = std::chrono::duration<double>(now - ui.last_frame_time).count();
    ui.last_frame_time = now;
    ui.total_time += ui.dt;
}
auto frame_indices(AppUI const &ui) -> std::pair<u32, u32> {
    const auto bounded_frame_index = static_cast<u32>(ui.frame_index % frames_in_flight);
    const auto last_frame_index = static_cast<u32>((ui.frame_index + frames_in_flight - 1u) % frames_in_flight);
    return {bounded_frame_index, last_frame_index};
}
auto handle_bindless_repopulation(AppContext &ctx, ResizeGraph &window_resize_graph) -> void {
    const u64 completed_now = std::min(ctx.gpu.tl_compute.completed, ctx.gpu.tl_graphics.completed);

    if (!ctx.gpu.bindless.repopulate_if_needed(ctx.gpu.ctx.textures, ctx.gpu.ctx.samplers,
                                               ctx.gpu.ctx.comparison_samplers)) {
        return;
    }

    const VkExtent2D we =
            sanitize_window_extent(current_extent(ctx.gpu.window), ctx.gpu.physical_device, ctx.gpu.surface);

    if (we.width != 0 && we.height != 0) {
        window_resize_graph.rebuild(we, ResizeContext{.ctx = ctx.gpu.ctx, .retire_value = completed_now},
                                    ResizeTrigger::Shaders);
    }

    info("Bindless set was repopulated, resizing pipelines.");
}
auto update_pending_resize(AppUI &ui, VkExtent2D desired_scene_extent) -> void {
    auto &pr = ui.pending_resize;

    const bool desired_changed = (desired_scene_extent.width != ui.last_viewport_extent.width) ||
                                 (desired_scene_extent.height != ui.last_viewport_extent.height);

    if (!desired_changed) {
        return;
    }

    if (!pr.has || pr.desired.width != desired_scene_extent.width || pr.desired.height != desired_scene_extent.height) {
        pr.desired = desired_scene_extent;
        pr.has = true;
        pr.last_change_time_s = ui.total_time;
    }
}

auto commit_resizes(AppContext &ctx, ResizeGraph &window_resize_graph, ResizeGraph &scene_resize_graph,
                    VkExtent2D window_extent, VkExtent2D &last_window_extent, VkExtent2D &render_scene_extent) -> bool {
    const u64 safe_retire = max_in_flight_retire_value(ctx.res);

    // --- window resize / shader triggers ---
    const bool window_resized =
            (window_extent.width != last_window_extent.width) || (window_extent.height != last_window_extent.height);

    const ResizeTrigger window_manual = window_resize_graph.get_and_clear_triggers();

    if (window_resized || window_manual != ResizeTrigger::Empty) {
        ResizeTrigger trigger = window_manual;
        if (window_resized) {
            trigger = trigger | ResizeTrigger::Extent;
        }

        window_resize_graph.rebuild(window_extent, ResizeContext{.ctx = ctx.gpu.ctx, .retire_value = safe_retire},
                                    trigger);
        last_window_extent = window_extent;
        return true; // skip rest of frame
    }

    // Pull scene manual triggers once
    ResizeTrigger scene_manual = scene_resize_graph.get_and_clear_triggers();

    // --- scene resize debounce ---
    auto &pr = ctx.ui.pending_resize;

    const bool lmb_down = ImGui::IsMouseDown(ImGuiMouseButton_Left);

    constexpr double debounce_s = 0.08;
    const bool stabilized = pr.has && ((ctx.ui.total_time - pr.last_change_time_s) > debounce_s);
    const bool released = pr.was_down && !lmb_down;
    const bool commit_now = pr.has && (released || stabilized);

    pr.was_down = lmb_down;

    // Build combined scene trigger set
    ResizeTrigger scene_trigger = scene_manual;
    VkExtent2D target_extent = render_scene_extent;

    if (commit_now) {
        scene_trigger = scene_trigger | ResizeTrigger::Extent;
        target_extent = pr.desired;
    }

    if (scene_trigger != ResizeTrigger::Empty) {
        scene_resize_graph.rebuild(target_extent, ResizeContext{.ctx = ctx.gpu.ctx, .retire_value = safe_retire},
                                   scene_trigger);
    }

    if (commit_now) {
        ctx.ui.last_viewport_extent = pr.desired;
        render_scene_extent = pr.desired;
        pr.has = false;

        info("Committed new viewport extent: {}x{}", render_scene_extent.width, render_scene_extent.height);
    }

    return false;
}


auto choose_render_scene_extent(AppUI const &ui, VkExtent2D desired_scene_extent) -> VkExtent2D {
    // Must be <= attachments this frame.
    if (ui.last_viewport_extent.width != 0 && ui.last_viewport_extent.height != 0) {
        return ui.last_viewport_extent;
    }
    return desired_scene_extent;
}
