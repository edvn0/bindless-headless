#include "app/frame.hxx"

#include "app/app.hxx"

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
                                float z_near) -> void {
    const float aspect = static_cast<float>(extent.width) / std::max(1.0f, static_cast<float>(extent.height));

    ubo.view = cam.view_matrix();
    ubo.projection = PerspectiveRH_ReverseZ_Inf(fov_y_radians, aspect, z_near);
    ubo.inv_projection = glm::inverse(ubo.projection);
    ubo.view_projection = ubo.projection * ubo.view;
    ubo.camera_position = glm::vec4(cam.camera_position(), 1.0f);
    ubo.inv_view_projection = glm::inverse(ubo.view_projection);
    ubo.viewport_size = glm::vec2(static_cast<float>(extent.width), static_cast<float>(extent.height));

    const auto planes = extract_frustum_planes(ubo.inv_view_projection);
    ubo.frustum_planes = planes;
}

auto write_camera_to_frame_ubo(RenderContext &ctx, AlignedRingBuffer<FrameUBO> &frame_ubo_ring, u32 frame_index,
                               const EditorCamera &cam, VkExtent2D extent, float fov_y_radians, float z_near) -> void {
    FrameUBO ubo{};
    fill_frame_ubo_from_camera(ubo, cam, extent, fov_y_radians, z_near);

    frame_ubo_ring.write_field(ctx, frame_index, ubo.view, offsetof(FrameUBO, view));
    frame_ubo_ring.write_field(ctx, frame_index, ubo.projection, offsetof(FrameUBO, projection));
    frame_ubo_ring.write_field(ctx, frame_index, ubo.view_projection, offsetof(FrameUBO, view_projection));
    frame_ubo_ring.write_field(ctx, frame_index, ubo.inv_view_projection, offsetof(FrameUBO, inv_view_projection));
    frame_ubo_ring.write_field(ctx, frame_index, ubo.inv_projection, offsetof(FrameUBO, inv_projection));
    frame_ubo_ring.write_field(ctx, frame_index, ubo.camera_position, offsetof(FrameUBO, camera_position));
    frame_ubo_ring.write_field(ctx, frame_index, ubo.frustum_planes, offsetof(FrameUBO, frustum_planes));
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
    // returns true if caller should early-continue (e.g. after window rebuild)
    const u64 safe_retire = max_in_flight_retire_value(ctx.res);

    // --- window resize / shader triggers ---
    const bool window_resized =
            (window_extent.width != last_window_extent.width) || (window_extent.height != last_window_extent.height);

    ResizeTrigger manual_trigger = window_resize_graph.get_and_clear_triggers();

    if (window_resized || manual_trigger != ResizeTrigger::None) {
        ResizeTrigger trigger = manual_trigger;
        if (window_resized) {
            trigger = trigger | ResizeTrigger::Extent;
        }

        window_resize_graph.rebuild(window_extent, ResizeContext{.ctx = ctx.gpu.ctx, .retire_value = safe_retire},
                                    trigger);
        last_window_extent = window_extent;

        // Important: if swapchain/pipelines were rebuilt, skip the rest of this frame.
        return true;
    }

    // --- scene resize debounce ---
    auto &pr = ctx.ui.pending_resize;

    const bool lmb_down = ImGui::IsMouseDown(ImGuiMouseButton_Left);

    constexpr double debounce_s = 0.08;
    const bool stabilized = pr.has && ((ctx.ui.total_time - pr.last_change_time_s) > debounce_s);
    const bool released = pr.was_down && !lmb_down;
    const bool commit_now = pr.has && (released || stabilized);

    pr.was_down = lmb_down;

    if (commit_now) {
        scene_resize_graph.rebuild(pr.desired, ResizeContext{.ctx = ctx.gpu.ctx, .retire_value = safe_retire},
                                   ResizeTrigger::Extent);

        ctx.ui.last_viewport_extent = pr.desired;
        render_scene_extent = pr.desired; // now safe to use immediately
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
