// app/ui.hxx
#include "app/ui.hxx"

#include "app/app.hxx"

#include "BindlessHeadless.hxx"
#include "FrameQuery.hxx"
#include "Pool.hxx"
#include "RenderContext.hxx"
#include "imgui.h"

#include <algorithm>
#include <array>
#include <ranges>
#include <string_view>

static constexpr auto widget = [](const std::string_view name, auto &&func) {
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize;
    ImGui::Begin(name.data(), nullptr, flags);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6, 6));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 4));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 2));

    func();

    ImGui::PopStyleVar(3);
    ImGui::End();
};

auto draw_ui(AppContext &ctx, u32 frame_index, AppState &output) -> void {
    // ---- Dockspace root ----
    ImGuiViewport *main_vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(main_vp->WorkPos);
    ImGui::SetNextWindowSize(main_vp->WorkSize);
    ImGui::SetNextWindowViewport(main_vp->ID);

    ImGuiWindowFlags dock_flags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
                                  ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                                  ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));

    ImGui::Begin("DockSpaceWindow", nullptr, dock_flags);
    ImGui::PopStyleVar(5);

    ImGuiID dockspace_id = ImGui::GetID("MainDockSpace");
    ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_NoWindowMenuButton | ImGuiDockNodeFlags_NoCloseButton;
    ImGui::DockSpace(dockspace_id, ImVec2(0, 0), dockspace_flags);
    ImGui::End();

    constexpr ImGuiWindowFlags viewport_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
    ImGui::Begin("Viewport", nullptr, viewport_flags);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(0, 0));

    {
        output.viewport_input = {};

        ImVec2 avail = ImGui::GetContentRegionAvail();
        if (avail.x >= 1.0f && avail.y >= 1.0f) {
            ImGui::InvisibleButton("##scene_viewport_hitbox", avail,
                                   ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight |
                                           ImGuiButtonFlags_MouseButtonMiddle);

            const ImGuiID viewport_id = ImGui::GetItemID();
            const ImGuiID hovered_id = ImGui::GetHoveredID();
            const ImGuiID active_id = ImGui::GetActiveID();

            const ImVec2 p0 = ImGui::GetItemRectMin();
            const ImVec2 p1 = ImGui::GetItemRectMax();

            ImGui::GetWindowDrawList()->AddImage(ImTextureID{ctx.res.tonemapped.index()}, p0, p1);

            output.viewport_input.min = p0;
            output.viewport_input.max = p1;

            output.viewport_input.hovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
            output.viewport_input.focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);

            output.viewport_input.viewport_item_id = viewport_id;
            output.viewport_input.hovered_id = hovered_id;
            output.viewport_input.active_id = active_id;

            const bool other_hovered = (hovered_id != 0 && hovered_id != viewport_id);
            const bool other_active = (active_id != 0 && active_id != viewport_id);

            output.viewport_input.imgui_blocks_mouse = other_hovered || other_active;

            output.viewport_input.imgui_blocks_keyboard = other_active;
        }
    }

    ImGui::PopStyleVar(4);
    ImGui::End();

    // ---- Queries / stats collection ----
    auto compute_res = read_timestamp_pairs_ms(ctx.gpu.ctx, ctx.pipes.compute_query_pool[frame_index]);
    auto c_stats = read_compute_stats(ctx.gpu.ctx, ctx.pipes.compute_stats_pool[frame_index]);
    auto graphics_res = read_timestamp_pairs_ms(ctx.gpu.ctx, ctx.pipes.graphics_query_pool[frame_index]);
    auto g_stats = read_graphics_stats(ctx.gpu.ctx, ctx.pipes.graphics_stats_pool[frame_index]);

    if (compute_res.has_value()) {
        const auto &c_times = *compute_res;
        ctx.ui.gpu_frame_graph.push_sample(0, c_times[static_cast<u32>(ComputeIndex::Rotate)]);
        ctx.ui.gpu_frame_graph.push_sample(1, c_times[static_cast<u32>(ComputeIndex::Cull)]);
        ctx.ui.gpu_frame_graph.push_sample(2, c_times[static_cast<u32>(ComputeIndex::Clustering)]);
    }

    if (graphics_res.has_value()) {
        const auto &g_times = *graphics_res;
        ctx.ui.gpu_frame_graph.push_sample(3, g_times[static_cast<u32>(GraphicsIndex::PreDepth)]);
        ctx.ui.gpu_frame_graph.push_sample(4, g_times[static_cast<u32>(GraphicsIndex::GBuffer)]);
        ctx.ui.gpu_frame_graph.push_sample(5, g_times[static_cast<u32>(GraphicsIndex::Deferred)]);
        ctx.ui.gpu_frame_graph.push_sample(6, g_times[static_cast<u32>(GraphicsIndex::Tonemap)]);
        ctx.ui.gpu_frame_graph.push_sample(7, g_times[static_cast<u32>(GraphicsIndex::Present)]);
        ctx.ui.gpu_frame_graph.push_sample(8, g_times[static_cast<u32>(GraphicsIndex::ShadowMap)]);
    }

    widget("Performance Graphs", [&] {
        static int view_mode = 0;
        static bool shared_scale = false;

        ImGui::RadioButton("Combined View", &view_mode, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Split View", &view_mode, 1);

        if (view_mode == 1) {
            ImGui::SameLine();
            ImGui::Checkbox("Shared Scale", &shared_scale);
        }

        ImGui::Separator();

        if (view_mode == 0) {
            ctx.ui.gpu_frame_graph.render("GPU Frame Times", ImVec2(0, 200));
        } else {
            ctx.ui.gpu_frame_graph.render_split("GPU", ImVec2(-1, 80), shared_scale);
        }
    });

    widget("Sun & Shadow Settings", [&] {
        if (ImGui::CollapsingHeader("Sun Direction", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Elevation", &ctx.ui.sun_config.elevation_degrees, 0.0f, 90.0f, "%.1f°");
            ImGui::SliderFloat("Azimuth", &ctx.ui.sun_config.azimuth_degrees, 0.0f, 360.0f, "%.1f°");
            ImGui::SliderFloat("Intensity", &ctx.ui.sun_config.intensity, 0.0f, 5.0f, "%.2f");

            // Optional: preset buttons
            if (ImGui::Button("Morning (East)")) {
                ctx.ui.sun_config.elevation_degrees = 30.0f;
                ctx.ui.sun_config.azimuth_degrees = 90.0f;
            }
            ImGui::SameLine();
            if (ImGui::Button("Noon (Overhead)")) {
                ctx.ui.sun_config.elevation_degrees = 80.0f;
                ctx.ui.sun_config.azimuth_degrees = 0.0f;
            }
            ImGui::SameLine();
            if (ImGui::Button("Sunset (West)")) {
                ctx.ui.sun_config.elevation_degrees = 10.0f;
                ctx.ui.sun_config.azimuth_degrees = 270.0f;
            }
        }

        if (ImGui::CollapsingHeader("Shadow Settings")) {
            ImGui::SliderFloat("Shadow Distance", &ctx.ui.shadow_config.shadow_distance, -20000.0f, 20000.0f);
            ImGui::SliderFloat("Ortho Size", &ctx.ui.shadow_config.ortho_size, 5.0f, 10000.0f);
            ImGui::SliderFloat("Near Plane", &ctx.ui.shadow_config.near_plane, -50000.F, 50000.0F);
            ImGui::SliderFloat("Far Plane", &ctx.ui.shadow_config.far_plane, -50000.0F, 50000.0f);
            ImGui::DragFloat3("Light Target", &ctx.ui.shadow_config.light_target.x, 0.1f);

            ImGui::ImageButton("Shadow map", ImTextureRef{ctx.res.directional_shadow_map_depth.index()},
                               {
                                       ImGui::GetContentRegionAvail().y,
                                       ImGui::GetContentRegionAvail().y,
                               });
        }
    });


    static u64 total_frame_counter = 0;

    widget("Frame Profile", [&] {
        ImGui::Text("Frame Profile [#%llu]", total_frame_counter++);
        ImGui::Separator();

        if (compute_res.has_value()) {
            if (ImGui::CollapsingHeader("Compute Phases", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::BeginTable("ComputeTable", 2,
                                      ImGuiTableFlags_BordersInner | ImGuiTableFlags_RowBg |
                                              ImGuiTableFlags_SizingFixedFit)) {
                    ImGui::TableSetupColumn("Phase");
                    ImGui::TableSetupColumn("Time (ms)");
                    ImGui::TableHeadersRow();

                    const auto &t = *compute_res;

                    auto row_c = [&](const char *name, ComputeIndex idx) {
                        u32 i = static_cast<u32>(idx);
                        if (i >= t.size()) {
                            return;
                        }

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted(name);

                        ImGui::TableNextColumn();
                        ImGui::Text("%.4f", t[i]);

                        if (c_stats.has_value() && i < c_stats->size()) {
                            ImGui::TableNextRow();
                            ImGui::TableNextColumn();
                            ImGui::Indent();
                            ImGui::Text("Invocations:");
                            ImGui::Unindent();

                            ImGui::TableNextColumn();
                            ImGui::Text("%llu", (*c_stats)[i].compute_shader_invocations);
                        }
                    };

                    row_c("Rotate", ComputeIndex::Rotate);
                    row_c("Cull", ComputeIndex::Cull);
                    row_c("Clustering", ComputeIndex::Clustering);

                    ImGui::EndTable();
                }
            }
        }

        if (graphics_res.has_value()) {
            ImGui::Separator();

            if (ImGui::CollapsingHeader("Graphics Phases", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::BeginTable("GraphicsTable", 2,
                                      ImGuiTableFlags_BordersInner | ImGuiTableFlags_RowBg |
                                              ImGuiTableFlags_SizingFixedFit)) {
                    ImGui::TableSetupColumn("Phase");
                    ImGui::TableSetupColumn("Time (ms)");
                    ImGui::TableHeadersRow();

                    const auto &t = *graphics_res;

                    auto row_g = [&](const char *name, GraphicsIndex idx) {
                        u32 i = static_cast<u32>(idx);
                        if (i >= t.size()) {
                            return;
                        }

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted(name);

                        ImGui::TableNextColumn();
                        ImGui::Text("%.4f", t[i]);
                    };

                    row_g("Pre-Depth", GraphicsIndex::PreDepth);
                    row_g("GBuffer", GraphicsIndex::GBuffer);
                    row_g("Deferred", GraphicsIndex::Deferred);
                    row_g("Tonemap", GraphicsIndex::Tonemap);
                    row_g("Present", GraphicsIndex::Present);
                    row_g("ShadowMap", GraphicsIndex::ShadowMap);

                    ImGui::EndTable();
                }

                if (g_stats.has_value()) {
                    ImGui::Separator();
                    ImGui::Text("Geometry Totals");

                    const auto &gb = (*g_stats)[static_cast<u32>(GraphicsIndex::GBuffer)];

                    ImGui::BulletText("Vertices: %llu", gb.input_assembly_vertices);
                    ImGui::BulletText("Primitives: %llu", gb.input_assembly_primitives);
                    ImGui::BulletText("Fragment Invocations: %llu", gb.fragment_shader_invocations);
                }
            }
        }

        if (compute_res.has_value() && graphics_res.has_value()) {
            const auto &c_times = *compute_res;
            const auto &g_times = *graphics_res;

            double total_ms = 0.0;
            for (double m: c_times)
                total_ms += m;
            for (double m: g_times)
                total_ms += m;

            double clustering_ms = c_times[static_cast<u32>(ComputeIndex::Clustering)];
            double clustering_pct = (total_ms > 0.0) ? (clustering_ms / total_ms) * 100.0 : 0.0;

            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.9f, 0.8f, 0.3f, 1.0f), "Clustering is %.1f%% of GPU frame time",
                               clustering_pct);
        }
    });
}

auto run_ui_frame(AppContext &ctx) -> UiFrameResult {
    UiFrameResult out{};

    const VkExtent2D raw_window_extent = current_extent(ctx.gpu.window);
    out.window_extent = sanitize_window_extent(raw_window_extent, ctx.gpu.physical_device, ctx.gpu.surface);

    if (out.window_extent.width == 0 || out.window_extent.height == 0) {
        out.minimized = true;
        return out;
    }

    ctx.ui.gui->begin_frame(ImGuiFramebuffer(out.window_extent, ctx.gpu.swapchain.format(),
                                             ctx.gpu.ctx.texture_format(ctx.res.tonemapped),
                                             ctx.gpu.swapchain.color_space()));

    draw_ui(ctx, static_cast<u32>(ctx.ui.frame_index % frames_in_flight), ctx.ui.app_state);

    ctx.ui.gui->end_frame();

    out.desired_scene_extent =
            sanitize_scene_extent(ctx.ui.app_state.viewport_input.extent(),
                                  (ctx.ui.last_viewport_extent.width == 0 || ctx.ui.last_viewport_extent.height == 0)
                                          ? VkExtent2D{ctx.gpu.opts->width, ctx.gpu.opts->height}
                                          : ctx.ui.last_viewport_extent,
                                  ctx.gpu.physical_device, ExtentBounds{.min_dim = 1, .max_dim = 4096});

    return out;
}
auto window_center(GLFWwindow *w) -> glm::vec2 {
    int ww = 0, wh = 0;
    glfwGetWindowSize(w, &ww, &wh);
    return glm::vec2{ww * 0.5f, wh * 0.5f};
}
auto begin_cursor_capture(GLFWwindow *w, AppState &app) -> void {
    if (glfwRawMouseMotionSupported()) {
        glfwSetInputMode(w, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }

    glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    app.cursor_captured = true;

    app.warp_center = window_center(w);
    app.warp_in_progress = true;
    glfwSetCursorPos(w, app.warp_center.x, app.warp_center.y);

    // Seed delta tracking at center
    app.last_mouse = app.warp_center;
    app.mouse_inited = true;
}
auto end_cursor_capture(GLFWwindow *w, AppState &app) -> void {
    if (glfwRawMouseMotionSupported()) {
        glfwSetInputMode(w, GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
    }

    glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    app.cursor_captured = false;
    app.warp_in_progress = false;
    app.mouse_inited = false; // force re-init when returning to normal mouse
}
