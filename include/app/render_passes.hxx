#pragma once

#include "app/app.hxx"

auto run_rotation_pass(AppContext &ctx, const u32 bounded_frame_index, const u32 last_frame_index,
                       const DeviceAddress &point_lights_base_addr, const SubmitSynchronisation &sync = no_waits)
        -> u64;

auto run_predepth_pass(AppContext &ctx, VkExtent2D frame_extent, const DrawRanges &ranges,
                       const u32 bounded_frame_index, const SubmitSynchronisation &sync) -> u64;

auto run_light_frustum_cull_pass(AppContext &ctx, const u32 bounded_frame_index, DeviceAddresses<4> &&device_addresses,
                                 const SubmitSynchronisation &sync) -> u64;

auto run_light_clustering_pass(AppContext &ctx, const u32 bounded_frame_index, DeviceAddresses<4> &&device_addresses,
                               const SubmitSynchronisation &sync) -> u64;

auto run_gbuffer_pass(AppContext &ctx, const VkExtent2D frame_extent, const DrawRanges &ranges,
                      const u32 bounded_frame_index, const SubmitSynchronisation &sync) -> u64;

auto run_deferred_lighting_pass(AppContext &ctx, const VkExtent2D frame_extent, DeviceAddresses<2> &&device_addresses,
                                const u32 bounded_frame_index, const SubmitSynchronisation &sync) -> u64;

auto run_tonemap_pass(AppContext &ctx, const VkExtent2D frame_extent, const u32 bounded_frame_index,
                      const SubmitSynchronisation &sync) -> u64;

auto run_imgui_pass(AppContext &ctx, const u32 bounded_frame_index, const SubmitSynchronisation &sync) -> u64;

auto run_swapchain_pass(AppContext &ctx, const u32 swap_image_index, const u32 bounded_frame_index,
                        const SubmitSynchronisation &sync) -> u64;
