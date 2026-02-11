#pragma once

#include "app/app.hxx"

using TimelineValue = u64;
using BoundedFrameIndex = u32;
using BoundedLastFrameIndex = u32;

auto run_rotation_pass(AppContext &, BoundedFrameIndex, BoundedLastFrameIndex,
                       const DeviceAddress &point_lights_base_addr, const SubmitSynchronisation &sync = no_waits)
        -> TimelineValue;

auto run_predepth_pass(AppContext &, VkExtent2D frame_extent, const DrawRanges &ranges, BoundedFrameIndex,
                       const SubmitSynchronisation &sync) -> TimelineValue;

auto run_light_frustum_cull_pass(AppContext &, BoundedFrameIndex, DeviceAddresses<4> &&device_addresses,
                                 const SubmitSynchronisation &sync) -> TimelineValue;

auto run_light_clustering_pass(AppContext &, BoundedFrameIndex, DeviceAddresses<4> &&device_addresses,
                               const SubmitSynchronisation &sync) -> TimelineValue;

auto run_directional_shadow_map_pass(AppContext &, const DrawRanges &ranges, BoundedFrameIndex,
                                     const SubmitSynchronisation &sync) -> TimelineValue;

auto run_gbuffer_pass(AppContext &, const VkExtent2D frame_extent, const DrawRanges &ranges, BoundedFrameIndex,
                      const SubmitSynchronisation &sync) -> TimelineValue;

auto run_deferred_lighting_pass(AppContext &, const VkExtent2D frame_extent, DeviceAddresses<2> &&device_addresses,
                                BoundedFrameIndex, const SubmitSynchronisation &sync) -> TimelineValue;

auto run_tonemap_pass(AppContext &, const VkExtent2D frame_extent, BoundedFrameIndex, const SubmitSynchronisation &sync)
        -> TimelineValue;

auto run_imgui_pass(AppContext &, BoundedFrameIndex, const SubmitSynchronisation &sync) -> TimelineValue;

auto run_swapchain_pass(AppContext &, const u32 swap_image_index, BoundedFrameIndex, const SubmitSynchronisation &sync)
        -> TimelineValue;
