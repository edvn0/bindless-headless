// app/ui.hxx
#include "app/ui.hxx"

#include "Types.hxx"
#include "app/app.hxx"

#include "BindlessHeadless.hxx"
#include "FrameQuery.hxx"
#include "Pool.hxx"
#include "RenderContext.hxx"
#include "imgui.h"
#include "ui/StyleGuard.hxx"

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


    {

        StyleGuard viewport_guard(std::pair{ImGuiStyleVar_WindowPadding, ImVec2(0, 0)},
                                  std::pair{ImGuiStyleVar_ItemSpacing, ImVec2(0, 0)},
                                  std::pair{ImGuiStyleVar_FramePadding, ImVec2(0, 0)},
                                  std::pair{ImGuiStyleVar_CellPadding, ImVec2(0, 0)});

        constexpr ImGuiWindowFlags viewport_flags =
                ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
        ImGui::Begin("Viewport", nullptr, viewport_flags);
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
    ImGui::End();


    // ---- Queries / stats collection ----
    auto compute_res = read_timestamp_pairs_ms(ctx.gpu.ctx, ctx.pipes.compute_query_pool[frame_index]);
    auto c_stats = read_compute_stats(ctx.gpu.ctx, ctx.pipes.compute_stats_pool[frame_index]);
    auto graphics_res = read_timestamp_pairs_ms(ctx.gpu.ctx, ctx.pipes.graphics_query_pool[frame_index]);
    auto g_stats = read_graphics_stats(ctx.gpu.ctx, ctx.pipes.graphics_stats_pool[frame_index]);

    auto index = 0;
    if (compute_res.has_value()) {
        const auto &c_times = *compute_res;
        for (size_t i = 0; i < compute_stages.size(); ++i) {
            ctx.ui.gpu_frame_graph.push_sample(static_cast<int>(index++), c_times[static_cast<u32>(compute_stages[i])]);
        }
    }

    if (graphics_res.has_value()) {
        const auto &g_times = *graphics_res;
        for (size_t i = 0; i < graphics_stages.size(); ++i) {
            ctx.ui.gpu_frame_graph.push_sample(static_cast<int>(index++),
                                               g_times[static_cast<u32>(graphics_stages[i])]);
        }
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

        StyleGuard guard(std::pair{ImGuiStyleVar_WindowPadding, ImVec2(0, 0)},
                         std::pair{ImGuiStyleVar_FramePadding, ImVec2(0, 0)});

        if (view_mode == 0) {
            ctx.ui.gpu_frame_graph.render("GPU Frame Times", ImVec2(-1, 200));
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

            ImGui::DragFloat("Depth bias constant factor", &ctx.ui.shadow_config.depth_bias_constant_factor);
            ImGui::DragFloat("Depth bias clamp", &ctx.ui.shadow_config.depth_bias_clamp);
            ImGui::DragFloat("Depth bias slope factor", &ctx.ui.shadow_config.depth_bias_slope_factor);

            ImGui::ImageButton("Shadow map", ImTextureRef{ctx.res.directional_shadow_map_depth.index()},
                               {
                                       ImGui::GetContentRegionAvail().y,
                                       ImGui::GetContentRegionAvail().y,
                               });
        }
    });


    static u64 total_frame_counter = 0;

    widget("Frame Profile", [&] {
        ImGui::Text("Frame Profile [#%lu]", total_frame_counter++);
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
                            ImGui::Text("%lu", (*c_stats)[i].compute_shader_invocations);
                        }
                    };

                    row_c("Rotate geometry", ComputeIndex::RotateGeometry);
                    row_c("Rotate lights", ComputeIndex::RotateLights);
                    row_c("Light Clustering", ComputeIndex::LightClustering);

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

                    ImGui::BulletText("Vertices: %lu", gb.input_assembly_vertices);
                    ImGui::BulletText("Primitives: %lu", gb.input_assembly_primitives);
                    ImGui::BulletText("Fragment Invocations: %lu", gb.fragment_shader_invocations);
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

            double clustering_ms = c_times[static_cast<u32>(ComputeIndex::LightClustering)];
            double clustering_pct = (total_ms > 0.0) ? (clustering_ms / total_ms) * 100.0 : 0.0;

            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.9f, 0.8f, 0.3f, 1.0f), "Clustering is %.1f%% of GPU frame time",
                               clustering_pct);
        }
    });

    widget("Render settings", [&] {
        /*    enum class ClusterDebugMode : u32 {
        None = 0,
        ClusterGrid = 1,
        LightCount = 2,
        LightDensity = 3,
        ClusterIndex = 4,
        DepthSlices = 5,
        LightHeatmap = 6,
        FirstLight = 7,
        ClusterOccupancy = 8,
    };*/
        /* ImGui::Combo("Cluster Debug Mode", reinterpret_cast<int *>(&ctx.ui.debug_mode),
                     "None\0Cluster Grid\0Light Count\0Light Density\0Cluster Index\0Depth Slices\0Light Heatmap\0First
        Light\0Cluster Occupancy\0");

           static constexpr std::array shadow_map_res_options = {512u, 1024u, 2048u, 4096u, 8192u};
        static int current_res_idx = 2; // Default to 2048
        if (ImGui::Combo("Shadow Map Resolution", &current_res_idx,
                         "512\01024\02048\04096\08192\0")) {
            ctx.ui.shadow_map_resolution = shadow_map_res_options[current_res_idx];
        } */

        const auto &preview_debug_mode = ctx.ui.debug_mode;
        if (ImGui::BeginCombo("Cluster Debug Mode", std::format("{}", static_cast<u32>(preview_debug_mode)).c_str(),
                              ImGuiComboFlags_HeightLarge)) {
            for (int i = 0; i < static_cast<int>(AppUI::ClusterDebugMode::Count); i++) {
                auto mode = static_cast<AppUI::ClusterDebugMode>(i);
                const char *mode_name = nullptr;
                switch (mode) {
                    using enum AppUI::ClusterDebugMode;
                    case None:
                        mode_name = "None";
                        break;
                    case ClusterGrid:
                        mode_name = "Cluster Grid";
                        break;
                    case LightCount:
                        mode_name = "Light Count";
                        break;
                    case LightDensity:
                        mode_name = "Light Density";
                        break;
                    case ClusterIndex:
                        mode_name = "Cluster Index";
                        break;
                    case DepthSlices:
                        mode_name = "Depth Slices";
                        break;
                    case LightHeatmap:
                        mode_name = "Light Heatmap";
                        break;
                    case FirstLight:
                        mode_name = "First Light";
                        break;
                    case ClusterOccupancy:
                        mode_name = "Cluster Occupancy";
                        break;
                    default: {
                        continue;
                    }
                }

                if (ImGui::Selectable(mode_name, ctx.ui.debug_mode == mode)) {
                    ctx.ui.debug_mode = mode;
                }
            }
            ImGui::EndCombo();
        }

        const auto &preview_value = ctx.ui.shadow_map_resolution.peek();
        if (ImGui::BeginCombo("Shadow Map Resolution", std::format("{}x{}", preview_value, preview_value).c_str(),
                              ImGuiComboFlags_HeightLarge)) {
            static constexpr std::array shadow_map_res_options = {512u, 1024u, 2048u, 4096u, 8192u};
            for (size_t i = 0; i < shadow_map_res_options.size(); i++) {
                const auto res = shadow_map_res_options[i];
                auto label = std::format("{}x{}", res, res);

                if (ImGui::Selectable(label.c_str(), ctx.ui.shadow_map_resolution.peek() == res)) {
                    ctx.ui.shadow_map_resolution = res;
                    ctx.gpu.scene_resize_graph.trigger_resize(ResizeTrigger::ShadowMap);
                }
            }
            ImGui::EndCombo();
        }
    });

    widget("Debug clustering", [&c = ctx] {
        ImGui::ImageButton("Clustering", ImTextureRef{c.res.debug_culling.index()},
                           {
                                   ImGui::GetContentRegionAvail().x,
                                   ImGui::GetContentRegionAvail().y,
                           });
    });

    widget("Cluster Configuration", [&] {
    auto& latch = ctx.ui.clustering_config;
    
    // The "pending" state exists only within the UI
    static ClusterConfig pending = latch.peek();
    static bool is_dirty = false;

    if (ImGui::CollapsingHeader("Grid Dimensions", ImGuiTreeNodeFlags_DefaultOpen)) {
        is_dirty |= ImGui::DragScalar("Tiles X", ImGuiDataType_U32, &pending.tiles_x, 1.0f, nullptr, nullptr, "%u");
        is_dirty |= ImGui::DragScalar("Tiles Y", ImGuiDataType_U32, &pending.tiles_y, 1.0f, nullptr, nullptr, "%u");
        is_dirty |= ImGui::DragScalar("Tiles Z", ImGuiDataType_U32, &pending.tiles_z, 1.0f, nullptr, nullptr, "%u");
    }

    if (ImGui::CollapsingHeader("Frustum Settings")) {
        is_dirty |= ImGui::SliderFloat("Z Near", &pending.z_near, 0.1f, 10.0f);
        is_dirty |= ImGui::SliderFloat("Z Far", &pending.z_far, 10.0f, 10000.0f);
    }

    ImGui::Separator();

    // Feedback on what the Apply button will actually do
    const u32 total_clusters = pending.tiles_x * pending.tiles_y * pending.tiles_z;
    ImGui::Text("Pending Clusters: %u", total_clusters);

    if (is_dirty) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.4f, 0.2f, 1.0f));
        if (ImGui::Button("Apply Changes")) {
            latch = pending; // Push to the latch
            ctx.gpu.scene_resize_graph.trigger_resize(ResizeTrigger::Clustering);
            is_dirty = false;
        }
        ImGui::PopStyleColor();
        
        ImGui::SameLine();
        
        if (ImGui::Button("Clear")) {
            pending = latch.peek(); // Reset to current engine state
            is_dirty = false;
        }
    } else {
        ImGui::BeginDisabled();
        ImGui::Button("Up to date");
        ImGui::EndDisabled();
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

    static u8 warmup_frames = frames_in_flight;
    if (warmup_frames > 0) [[unlikely]] {
        --warmup_frames;
    } else {
        u32 index = static_cast<u32>(ctx.ui.frame_index % frames_in_flight);
        draw_ui(ctx, index, ctx.ui.app_state);
    }
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
    double x{}, y{};
    glfwGetCursorPos(w, &x, &y);
    app.last_mouse = glm::vec2(static_cast<float>(x), static_cast<float>(y));

    if (glfwRawMouseMotionSupported()) {
        glfwSetInputMode(w, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }

    glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    app.cursor_captured = true;
    app.mouse_inited = true;
    app.warp_in_progress = false; // Usually not needed in DISABLED mode
}

auto end_cursor_capture(GLFWwindow *w, AppState &app) -> void {
    if (glfwRawMouseMotionSupported()) {
        glfwSetInputMode(w, GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
    }

    glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    app.cursor_captured = false;
    app.mouse_inited = false;
}
