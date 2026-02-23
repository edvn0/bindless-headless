#pragma once

#include "app/app.hxx"

using TimelineValue = u64;
using BoundedFrameIndex = u32;
using BoundedLastFrameIndex = u32;

auto run_rotation_pass(AppContext &, BoundedFrameIndex, BoundedLastFrameIndex,
                       const DeviceAddress &point_lights_base_addr, const SubmitSynchronisation &sync = no_waits)
        -> TimelineValue;

auto run_predepth_pass(AppContext &ctx, VkExtent2D frame_extent,
                       std::span<const MeshInstanceRange> mesh_instance_ranges,
                       std::span<const DrawRanges> ranges,
                       u32 bounded_frame_index,
                       const SubmitSynchronisation &sync) -> u64;

auto run_light_clustering_pass(AppContext &, BoundedFrameIndex,
                               const SubmitSynchronisation &sync) -> TimelineValue;

auto run_directional_shadow_map_pass(AppContext &ctx,
                                     std::span<const MeshInstanceRange> mesh_instance_ranges,
                                     std::span<const DrawRanges> ranges,
                                     u32 bounded_frame_index,
                                     const SubmitSynchronisation &sync) -> u64;

auto run_gbuffer_pass(AppContext &ctx, VkExtent2D frame_extent,
                      std::span<const MeshInstanceRange> mesh_instance_ranges,
                      std::span<const DrawRanges> ranges,
                      u32 bounded_frame_index,
                      const SubmitSynchronisation &sync) -> u64;

auto run_deferred_lighting_pass(AppContext &, const VkExtent2D frame_extent, u32 light_slot,
                                BoundedFrameIndex, const SubmitSynchronisation &sync) -> TimelineValue;

auto run_environment_skybox_pass(AppContext &ctx, VkExtent2D frame_extent, BoundedFrameIndex, const SubmitSynchronisation &sync)
        -> TimelineValue;

auto run_tonemap_pass(AppContext &, const VkExtent2D frame_extent, BoundedFrameIndex, const SubmitSynchronisation &sync)
        -> TimelineValue;

auto run_imgui_pass(AppContext &, BoundedFrameIndex, const SubmitSynchronisation &sync) -> TimelineValue;

auto run_swapchain_pass(AppContext &, const u32 swap_image_index, BoundedFrameIndex, const SubmitSynchronisation &sync)
        -> TimelineValue;
