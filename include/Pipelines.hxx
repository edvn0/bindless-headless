#pragma once

#include <volk.h>

#include "Forward.hxx"
#include "Types.hxx"

#include <array>
#include <ranges>
#include <string_view>
#include <vector>

inline constexpr u32 THREADS_PER_GROUP = 64;
inline constexpr u32 MAX_WAVES_PER_GROUP = 4;

struct PointLightCullingPushConstants {
    const DeviceAddress ubo;
    const DeviceAddress lights;
    const DeviceAddress flags;
    const DeviceAddress prefix;
    const DeviceAddress compact;
    const DeviceAddress culled_light_count; // OUTPUT
    const u32 light_count;
};

/**
* struct PC {
    UBO* frame_ubo;
    Transform* transforms;
    uint* draw_material_ids;
    Material* materials;
    uint       base_draw_id;
    uint       sampler_index;
};
 */
struct PredepthPushConstants {
    const DeviceAddress ubo;
    const DeviceAddress transforms;
    const DeviceAddress draw_material_ids;
    const DeviceAddress materials;
     u32 base_draw_id;
    const u32 sampler_index {0};
};

struct RenderingPushConstants {
    const DeviceAddress ubo;
    const DeviceAddress transforms;
    const DeviceAddress draw_material_ids;
    const DeviceAddress materials;
     u32 base_draw_id;
    const u32 sampler_index;
};

struct TonemapPushConstants {
    float exposure;
    const u32 image_index;
    const u32 sampler_index;
};

struct RotateCubesPushConstant {
    u32 cube_count;
    f32 delta_time;
    f32 rads_per_second;
    const DeviceAddress transforms;
    const DeviceAddress previous_frame_transforms;
};

struct DeferredLightingPushConstants {
    const DeviceAddress frame_ubo;
    const DeviceAddress point_lights;
    const DeviceAddress point_light_count;

    u32 gbuffer0_index;
    u32 gbuffer1_index;
    u32 gbuffer2_index;
    u32 depth_index;
    u32 lit_hdr_uav_index {0}; // For the compute version, just 0 always
    u32 sampler_index;       // probably linear_clamp
};

struct CompiledPipeline {
    VkPipeline pipeline{VK_NULL_HANDLE};
    VkPipelineLayout layout{VK_NULL_HANDLE};
};

auto create_compute_pipeline(VkDevice, PipelineCache &, VkDescriptorSetLayout, const std::vector<u32> &,
                             std::string_view) -> CompiledPipeline;

template<std::size_t N>
auto create_compute_pipelines(VkDevice device, PipelineCache &cache, VkDescriptorSetLayout layout,
                              std::span<std::vector<u32>, N> codes, std::span<const std::string_view, N> names)
        -> std::array<CompiledPipeline, N> {
    std::array<CompiledPipeline, N> out{};

    auto rng = std::views::zip(codes, names) | std::views::transform([&](auto &&zipped) {
                   auto &&[code, name] = zipped;
                   return create_compute_pipeline(device, cache, layout, code, name);
               });

    std::ranges::copy(rng, out.begin());
    return out;
}

// Pipelines.hxx additions (signatures)
auto create_gbuffer_pipeline(VkDevice device,
                             PipelineCache& cache,
                             VkDescriptorSetLayout bindless_layout,
                             const std::vector<u32>& vert_code,
                             const std::vector<u32>& frag_code,
                             VkFormat gbuffer0_format,
                             VkFormat gbuffer1_format,
                             VkFormat gbuffer2_format,
                             VkFormat depth_format) -> CompiledPipeline;

auto create_deferred_lighting_compute_pipeline(VkDevice device,
                                               PipelineCache& cache,
                                               VkDescriptorSetLayout bindless_layout,
                                               const std::vector<u32>& cs_code,
                                               std::string_view entry_name) -> CompiledPipeline;

auto create_deferred_lighting_graphics_pipeline(
        VkDevice device,
        PipelineCache &cache,
        VkDescriptorSetLayout bindless_layout,
        const std::vector<u32> &vert_code,
        const std::vector<u32> &frag_code,
        std::string_view vert_entry,
        std::string_view frag_entry,
        VkFormat color_format) -> CompiledPipeline;

auto create_predepth_pipeline(VkDevice, PipelineCache &, VkDescriptorSetLayout, const std::vector<uint32_t> &, VkFormat,
                              VkSampleCountFlagBits = VK_SAMPLE_COUNT_1_BIT) -> CompiledPipeline;
auto create_predepth_pipeline(VkDevice, PipelineCache &, VkDescriptorSetLayout, const std::vector<uint32_t> &, const std::vector<uint32_t> &, VkFormat,
                              VkSampleCountFlagBits = VK_SAMPLE_COUNT_1_BIT) -> CompiledPipeline;

auto create_tonemap_pipeline(VkDevice, PipelineCache &, VkDescriptorSetLayout, const std::vector<u32> &,
                             const std::vector<u32> &, const std::string_view, const std::string_view, VkFormat)
        -> CompiledPipeline;
