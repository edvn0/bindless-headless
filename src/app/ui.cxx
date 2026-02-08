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

auto draw_ui(PerformanceGraph<8, 120> &gpu_frame_graph, const RenderContext &ctx,
             std::span<QueryPoolHandle, frames_in_flight> compute_query_pool,
             std::span<QueryPoolHandle, frames_in_flight> compute_stats_pool,
             std::span<QueryPoolHandle, frames_in_flight> graphics_query_pool,
             std::span<QueryPoolHandle, frames_in_flight> graphics_stats_pool, uint32_t frame_index) -> void {

    auto compute_res = read_timestamp_pairs_ms(ctx, compute_query_pool[frame_index]);
    auto c_stats = read_compute_stats(ctx, compute_stats_pool[frame_index]);
    auto graphics_res = read_timestamp_pairs_ms(ctx, graphics_query_pool[frame_index]);
    auto g_stats = read_graphics_stats(ctx, graphics_stats_pool[frame_index]);

    if (compute_res.has_value()) {
        const auto &c_times = *compute_res;
        gpu_frame_graph.push_sample(0, c_times[static_cast<u32>(ComputeIndex::Rotate)]);
        gpu_frame_graph.push_sample(1, c_times[static_cast<u32>(ComputeIndex::Cull)]);
        gpu_frame_graph.push_sample(2, c_times[static_cast<u32>(ComputeIndex::Clustering)]);
    }

    if (graphics_res.has_value()) {
        const auto &g_times = *graphics_res;
        gpu_frame_graph.push_sample(3, g_times[static_cast<u32>(GraphicsIndex::PreDepth)]);
        gpu_frame_graph.push_sample(4, g_times[static_cast<u32>(GraphicsIndex::GBuffer)]);
        gpu_frame_graph.push_sample(5, g_times[static_cast<u32>(GraphicsIndex::Deferred)]);
        gpu_frame_graph.push_sample(6, g_times[static_cast<u32>(GraphicsIndex::Tonemap)]);
        gpu_frame_graph.push_sample(7, g_times[static_cast<u32>(GraphicsIndex::Present)]);
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
            gpu_frame_graph.render("GPU Frame Times", ImVec2(0, 200));
        } else {
            gpu_frame_graph.render_split("GPU", ImVec2(-1, 80), shared_scale);
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

    widget("Targets", [&] {
        using namespace std::string_view_literals;
        struct TargetView {
            const std::string_view name;
            const TextureHandle handle;
        };

        static std::array targets = {
                TargetView{.name = "GBuffer0 (Albedo/AO)"sv, .handle = gbuffer0_handle},
                TargetView{.name = "GBuffer1 (Normal/RM)"sv, .handle = gbuffer1_handle},
                TargetView{.name = "GBuffer2 (Emissive)"sv, .handle = gbuffer2_handle},
                TargetView{.name = "Depth"sv, .handle = depth_handle},
                TargetView{.name = "Culling Debug"sv, .handle = debug_culling_handle},
                TargetView{.name = "Lit HDR"sv, .handle = lit_hdr_handle},
                TargetView{.name = "Tonemapped"sv, .handle = tonemapped_target_handle},
        };

        if (ImGui::BeginTabBar("##TargetTabs", ImGuiTabBarFlags_None)) {
            for (auto &&[idx, target]: targets | std::views::enumerate) {
                ImGui::PushID(static_cast<i32>(idx));
                if (ImGui::BeginTabItem(target.name.data())) {
                    ImVec2 size = ImGui::GetContentRegionAvail();
                    ImGui::Image(ImTextureRef{ImTextureID{target.handle.index()}}, size);
                    ImGui::EndTabItem();
                }
                ImGui::PopID();
            }
            ImGui::EndTabBar();
        }
    });
#endif
}
