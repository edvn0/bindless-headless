#include "app/ui.hxx"

#include "BindlessHeadless.hxx"
#include "Pool.hxx"
#include "RenderContext.hxx"

#include "FrameQuery.hxx"

static constexpr auto widget = [](const std::string_view name, auto &&func) {
    ImGui::Begin(name.data());
    func();
    ImGui::End();
};

auto draw_ui(AppContext &ctx, u32 frame_index, VkExtent2D &output_extent) -> void {
    ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
    window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
    window_flags |= ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

    ImGui::Begin("DockSpaceWindow", nullptr, window_flags);
    ImGui::PopStyleVar(3);

    ImGuiID dockspace_id = ImGui::GetID("MainDockSpace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);

    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Performance Graphs");
            ImGui::MenuItem("Frame Profile");
            ImGui::MenuItem("Viewport");
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    ImGui::End();

    ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    {
        using namespace std::string_view_literals;
        struct TargetView {
            const std::string_view name;
            const TextureHandle handle;
        };

        std::array targets = {
                TargetView{.name = "GBuffer0 (Albedo/AO)"sv, .handle = ctx.res.gbuffer0},
                TargetView{.name = "GBuffer1 (Normal/RM)"sv, .handle = ctx.res.gbuffer1},
                TargetView{.name = "GBuffer2 (Emissive)"sv, .handle = ctx.res.gbuffer2},
                TargetView{.name = "Depth"sv, .handle = ctx.res.depth},
                TargetView{.name = "Culling Debug"sv, .handle = ctx.res.debug_culling},
                TargetView{.name = "Lit HDR"sv, .handle = ctx.res.lit_hdr},
                TargetView{.name = "Tonemapped"sv, .handle = ctx.res.tonemapped},
        };

        static int selected_target = 6; // Default to tonemapped

        ImGui::SetNextItemWidth(200.0f);
        if (ImGui::BeginCombo("##TargetSelect", targets[selected_target].name.data())) {
            for (auto &&[idx, target]: targets | std::views::enumerate) {
                const bool is_selected = (selected_target == static_cast<int>(idx));
                if (ImGui::Selectable(target.name.data(), is_selected)) {
                    selected_target = static_cast<int>(idx);
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        ImVec2 available_size = ImGui::GetContentRegionAvail();
        ImGui::Image(ImTextureRef{ImTextureID{targets[selected_target].handle.index()}}, available_size);
        output_extent.width = static_cast<u32>(available_size.x);
        output_extent.height = static_cast<u32>(available_size.y);
    }
    ImGui::End();

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

#ifdef ENABLED
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
                        if (i >= t.size())
                            return;

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
                        if (i >= t.size())
                            return;

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

                    ImGui::EndTable();
                }

                // Geometry totals
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
            double clustering_pct = (clustering_ms / total_ms) * 100.0;

            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.9f, 0.8f, 0.3f, 1.0f), "Clustering is %.1f%% of GPU frame time",
                               clustering_pct);
        }
    });
#endif
}
