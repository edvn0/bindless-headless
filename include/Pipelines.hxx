#include <volk.h>

#include "Forward.hxx"
#include "Types.hxx"

#include <array>
#include <ranges>
#include <string_view>
#include <vector>

static constexpr u32 THREADS_PER_GROUP = 64;
static constexpr u32 MAX_WAVES_PER_GROUP = 4;

struct PointLightCullingPushConstants {
    const DeviceAddress ubo;
    const DeviceAddress lights;
    const DeviceAddress flags;
    const DeviceAddress prefix;
    const DeviceAddress compact;
    const DeviceAddress culled_light_count; // OUTPUT
    const u32 light_count;
};

struct PredepthPushConstants {
    const DeviceAddress ubo;
    const DeviceAddress transforms;
    const DeviceAddress draw_material_ids;
    const u32 base_draw_id;
};

struct RenderingPushConstants {
    const DeviceAddress ubo;
    const DeviceAddress transforms;
    const DeviceAddress draw_material_ids;
    const DeviceAddress materials;
    const u32 base_draw_id;
};

struct TonemapPushConstants {
    float exposure;
    const u32 image_index;
    const u32 sampler_index;
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


auto create_predepth_pipeline(VkDevice, PipelineCache &, VkDescriptorSetLayout, const std::vector<uint32_t> &,
                              const std::vector<uint32_t> &, VkFormat, VkSampleCountFlagBits = VK_SAMPLE_COUNT_1_BIT)
        -> CompiledPipeline;

auto create_mesh_pipeline(VkDevice, PipelineCache &, VkDescriptorSetLayout, const std::vector<u32> &,
                          const std::vector<u32> &, VkFormat, VkSampleCountFlagBits = VK_SAMPLE_COUNT_1_BIT)
        -> CompiledPipeline;

auto create_tonemap_pipeline(VkDevice, PipelineCache &, VkDescriptorSetLayout, const std::vector<u32> &,
                             const std::vector<u32> &, const std::string_view, const std::string_view, VkFormat)
        -> CompiledPipeline;
