#pragma once

#include "app/app.hxx"

using TimelineValue = u64;
using BoundedFrameIndex = u32;
using BoundedLastFrameIndex = u32;

namespace RP {
    struct Markers {
        const QueryPoolState *ts{nullptr};
        const QueryPoolState *stats{nullptr};
    };

    struct FrameMarkers {
        Markers graphics{};
        Markers compute{};
    };

    auto setup_render_passes_for_frame(AppContext &, BoundedFrameIndex) -> void;
    auto get_frame_markers() -> FrameMarkers const &;

    struct Specification {
        u32 timestamp_begin{};
        u32 timestamp_end{};
        u32 stats_index{};
    };

    auto graphics_specification(GraphicsIndex idx) -> Specification;
    auto compute_specification(ComputeIndex idx) -> Specification;

    class Scope {
    public:
        Scope(VkCommandBuffer cmd, Markers m, Specification s,
              VkPipelineStageFlags2 ts_begin = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
              VkPipelineStageFlags2 ts_end = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);
        Scope(Scope const &) = delete;
        auto operator=(Scope const &) -> Scope & = delete;
        Scope(Scope &&) noexcept;
        ~Scope();

    private:
        VkCommandBuffer cmd_{};
        Markers m_{};
        Specification s_{};
        VkPipelineStageFlags2 ts_begin_{};
        VkPipelineStageFlags2 ts_end_{};
        bool active_{true};
    };

    inline auto begin_graphics(VkCommandBuffer cmd, GraphicsIndex idx,
                               VkPipelineStageFlags2 ts_begin = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                               VkPipelineStageFlags2 ts_end = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT) -> Scope {
        return {cmd, get_frame_markers().graphics, graphics_specification(idx), ts_begin, ts_end};
    }

    inline auto begin_compute(VkCommandBuffer cmd, ComputeIndex idx,
                              VkPipelineStageFlags2 ts_begin = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VkPipelineStageFlags2 ts_end = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT) -> Scope {
        return {cmd, get_frame_markers().compute, compute_specification(idx), ts_begin, ts_end};
    }

} // namespace RP


auto run_rotation_pass(AppContext &, BoundedFrameIndex, BoundedLastFrameIndex, const DeviceAddress &,
                       const SubmitSynchronisation & = no_waits) -> TimelineValue;

auto run_predepth_pass(AppContext &, VkExtent2D, std::span<const MeshInstanceRange>, std::span<const DrawRanges>, u32,
                       const SubmitSynchronisation &) -> u64;

auto run_light_clustering_pass(AppContext &, BoundedFrameIndex, const SubmitSynchronisation &) -> TimelineValue;

auto run_directional_shadow_map_pass(AppContext &, std::span<const MeshInstanceRange>, std::span<const DrawRanges>, u32,
                                     const SubmitSynchronisation &) -> u64;

auto run_gbuffer_pass(AppContext &, VkExtent2D, std::span<const MeshInstanceRange>, std::span<const DrawRanges>, u32,
                      const SubmitSynchronisation &) -> u64;

auto run_ssao_pass(AppContext &, VkExtent2D, BoundedFrameIndex, const SubmitSynchronisation &) -> TimelineValue;
auto run_ssao_blur_pass(AppContext &, VkExtent2D, const SubmitSynchronisation &) -> TimelineValue;

auto run_deferred_lighting_pass(AppContext &, const VkExtent2D, u32 light_slot, BoundedFrameIndex,
                                const SubmitSynchronisation &) -> TimelineValue;

auto run_environment_skybox_pass(AppContext &, VkExtent2D, BoundedFrameIndex, const SubmitSynchronisation &)
        -> TimelineValue;

auto run_bloom_pass(AppContext &, VkExtent2D, const SubmitSynchronisation &) -> TimelineValue;

auto run_billboard_pass(AppContext &, VkExtent2D, BoundedFrameIndex, const SubmitSynchronisation &) -> TimelineValue;

auto run_tonemap_pass(AppContext &, const VkExtent2D, BoundedFrameIndex, const SubmitSynchronisation &)
        -> TimelineValue;

auto run_imgui_pass(AppContext &, BoundedFrameIndex, const SubmitSynchronisation &) -> TimelineValue;

auto run_swapchain_pass(AppContext &, const u32 swap_image_index, BoundedFrameIndex, const SubmitSynchronisation &)
        -> TimelineValue;
