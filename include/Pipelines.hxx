#pragma once

#include <volk.h>

#include "Constants.hxx"
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

struct ClusteredLightCullingPushConstants {
    const DeviceAddress frame_ubo;
    const DeviceAddress all_lights;
    const DeviceAddress mesh_indirect;
    const DeviceAddress clusters;
    const DeviceAddress cluster_light_indices;

    float z_near;
    float z_far;
    float log_z_scale;

    u32 light_count;
    u32 tiles_x;
    u32 tiles_y;
    u32 tiles_z;
    u32 cluster_count;
};

struct PredepthPushConstants {
    const DeviceAddress ubo;
    const DeviceAddress transforms;
    const DeviceAddress draw_material_ids;
    const DeviceAddress materials;
    u32 base_draw_id;
    const u32 sampler_index{0};
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
    const u32   bloom_index;
    float bloom_strength;
};

struct RotatePushConstant {
    f32 delta_time;
    f32 rads_per_second;
    f32 total_time;
    u32 count;
    const DeviceAddress previous_frame_transforms;
    const DeviceAddress transforms;
    const DeviceAddress previous_point_lights;
    const DeviceAddress point_lights;
    const DeviceAddress static_point_lights;
};

struct DeferredLightingPushConstants {
    const DeviceAddress frame_ubo;
    const DeviceAddress point_lights; // All the lights
    const DeviceAddress clusters;
    const DeviceAddress cluster_light_indices;
    const glm::mat4 shadow_matrix;
    float log_z_scale;
    f32 near_plane{z_near};
    f32 far_plane {z_far};

    u32 tiles_x;
    u32 tiles_y;
    u32 tiles_z;
    u32 gbuffer0_index;
    u32 gbuffer1_index;
    u32 gbuffer2_index;
    u32 ssao_index;
    u32 depth_index;
    u32 sampler_index;
    u32 shadow_texture_index;
    u32 shadow_sampler_index;
    u32 debug_mode;
};

struct PresentPushConstants {
    u32 image_index;
    u32 sampler_index;
    u32 dst_is_srgb;
};

struct ShadowMapPushConstants {
    glm::mat4 light_view_proj;
    const DeviceAddress transforms;
    const DeviceAddress draw_material_ids;
    const DeviceAddress materials;
    u32 base_draw_id;
    const u32 sampler_index;
};

struct DebugClusteredPushConstants {
    const DeviceAddress frame_ubo;
    const DeviceAddress all_lights;
    const DeviceAddress clusters;
    const DeviceAddress cluster_light_indices;
    u32 tiles_x;
    u32 tiles_y;
    u32 tiles_z;
};

struct HeatmapPushConstants {
    const DeviceAddress clusters;
    u32 tiles_x;
    u32 tiles_y;
    u32 tiles_z;
    u32 max_lights_per_cluster; // for normalisation
    u32 debug_texture_uav_index;
    u32 cell_size;
    u32 slices_per_row;
};

struct SkyboxPushConstants {
    const DeviceAddress frame_ubo;
    u32 cubemap_index;
    u32 sampler_index;
};

struct SSAOPushConstants {
    const DeviceAddress frame_ubo;
    const DeviceAddress hemisphere_kernel;
    const DeviceAddress noise_kernel;
    u32 gbuffer0_index;
    u32 gbuffer1_index;
    u32 depth_index;
    u32 ssao_output_index;
    u32 sampler_index;
    f32 radius;
    f32 bias;
};

struct SSAOBlurPushConstants {
    u32 ssao_input_index;
    u32 ssao_output_index;
    u32 depth_index;
    u32 sampler_index;
    u32 horizontal;
};

struct BloomThresholdPushConstants {
    u32 src_index;       // lit_hdr bindless index
    u32 dst_index;       // bloom_threshold bindless UAV index
    u32 sampler_index;
    float threshold;     // luminance knee, ~1.0 for physical
    float knee;          // soft knee width
};

struct BloomDownsamplePushConstants {
    u32 src_index;
    u32 dst_index;
    u32 sampler_index;
    glm::vec2 src_texel_size;  // 1.0 / src_extent, avoids dynamic indexing in shader
};

struct BloomUpsamplePushConstants {
    u32 src_index;          // current level
    u32 accumulate_index;   // level below (being written into)
    u32 sampler_index;
    float filter_radius;    // tent radius in UV space, ~0.005
    float strength;         // blend weight on upsample accumulation
};

struct CompiledPipeline {
    VkPipeline pipeline{VK_NULL_HANDLE};
    VkPipelineLayout layout{VK_NULL_HANDLE};
};

auto create_compute_pipeline(VkDevice, PipelineCache *, VkDescriptorSetLayout, const std::vector<u32> &, std::size_t,
                             std::string_view) -> CompiledPipeline;

template<std::size_t N>
auto create_compute_pipelines(VkDevice device, PipelineCache *cache, VkDescriptorSetLayout layout,
                              std::optional<std::size_t> push_constant_size, std::span<std::vector<u32>, N> codes,
                              std::span<const std::string_view, N> names) -> std::array<CompiledPipeline, N> {
    std::array<CompiledPipeline, N> out{};

    // Sane default. Well, for now.
    auto chosen_size = push_constant_size.value_or(sizeof(PointLightCullingPushConstants));

    auto rng = std::views::zip(codes, names) | std::views::transform([&](auto &&zipped) {
                   auto &&[code, name] = zipped;
                   return create_compute_pipeline(device, cache, layout, code, chosen_size, name);
               });

    std::ranges::copy(rng, out.begin());
    return out;
}

auto create_gbuffer_pipeline(VkDevice device, PipelineCache *cache, VkDescriptorSetLayout bindless_layout,
                             const std::vector<u32> &vert_code, const std::vector<u32> &frag_code,
                             VkFormat gbuffer0_format, VkFormat gbuffer1_format, VkFormat gbuffer2_format,
                             VkFormat depth_format) -> CompiledPipeline;

auto create_deferred_lighting_graphics_pipeline(VkDevice device, PipelineCache *cache,
                                                VkDescriptorSetLayout bindless_layout, const std::vector<u32> &frag,
                                                VkShaderModule, std::string_view frag_entry, VkFormat color_format)
        -> CompiledPipeline;

auto create_predepth_pipeline(VkDevice, PipelineCache *, VkDescriptorSetLayout, const std::vector<u32> &, VkFormat,
                              VkSampleCountFlagBits = VK_SAMPLE_COUNT_1_BIT) -> CompiledPipeline;
auto create_predepth_pipeline(VkDevice, PipelineCache *, VkDescriptorSetLayout, const std::vector<u32> &,
                              const std::vector<u32> &, VkFormat, VkSampleCountFlagBits = VK_SAMPLE_COUNT_1_BIT)
        -> CompiledPipeline;

auto create_directional_shadow_map_pipeline(VkDevice device, PipelineCache *cache,
                                            VkDescriptorSetLayout bindless_layout,
                                            const std::vector<u32> &vert_code,
                                            const std::vector<u32> &frag_code, VkFormat depth_format,
                                            VkSampleCountFlagBits samples) -> CompiledPipeline;
auto create_directional_shadow_map_pipeline(VkDevice device, PipelineCache *cache,
                                            VkDescriptorSetLayout bindless_layout,
                                            const std::vector<u32> &vert_code, VkFormat depth_format,
                                            VkSampleCountFlagBits samples) -> CompiledPipeline;

auto create_tonemap_pipeline(VkDevice, PipelineCache *, VkDescriptorSetLayout, const std::vector<u32> &,
                             const std::vector<u32> &, const std::string_view, const std::string_view, VkFormat)
        -> CompiledPipeline;

auto create_light_volume_mesh_pipeline(VkDevice device, PipelineCache *cache, VkDescriptorSetLayout bindless_layout,
                                       const std::vector<u32> &task_code, const std::vector<u32> &mesh_code,
                                       const std::vector<u32> &frag_code, VkFormat color_format, VkFormat depth_format,
                                       VkSampleCountFlagBits samples) -> CompiledPipeline;


namespace Pipeline {
    enum class DepthMode {
        none, // No depth test/write (tonemap, deferred lighting)
        write, // Full depth write (predepth, shadow)
        test_equal, // Test only, equal (gbuffer after predepth)
        test_greater_equal, // Test only, >= reverse-Z (light volumes)
    };

    enum class CullMode {
        none,
        back,
        front,
    };

    struct ColorAttachmentInfo {
        VkFormat format {VK_FORMAT_UNDEFINED};
        bool blend_additive = false; // false = no blend, true = additive (light volumes)
    };

    struct VertexInputInfo {
        std::span<const VkVertexInputBindingDescription> bindings;
        std::span<const VkVertexInputAttributeDescription> attributes;
    };

    struct ShaderStageInfo {
        std::span<const u32> code;
        std::string_view entry;
        VkShaderStageFlagBits stage;
    };

    struct Graphics {
        VkDevice device;
        PipelineCache *cache;
        VkDescriptorSetLayout bindless_layout;
        std::string_view debug_name;

        std::span<const ShaderStageInfo> stages;

        u32 push_constant_size = 0;
        VkShaderStageFlags push_constant_stages = VK_SHADER_STAGE_ALL_GRAPHICS;

        std::span<const ColorAttachmentInfo> color_attachments; // empty = depth-only
        VkFormat depth_format = VK_FORMAT_UNDEFINED;

        DepthMode depth_mode = DepthMode::none;
        CullMode cull_mode = CullMode::none;
        bool depth_bias = false;

        std::optional<VertexInputInfo> vertex_input;

        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

        std::span<const VkDynamicState> extra_dynamic_states;
    };
    auto create_graphics_pipeline(const Graphics &info) -> CompiledPipeline;

    struct Fullscreen {
        VkDevice device;
        PipelineCache *cache;
        VkDescriptorSetLayout bindless_layout;

        VkShaderModule fullscreen_vs; // This is application cached.
        static constexpr std::string_view vs_entry{"main"};

        std::span<const u32> frag_code;
        std::string_view fs_entry;

        VkFormat color_format;

        u32 push_constant_size;
        VkShaderStageFlags push_constant_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        bool enable_blend = false;
    };
    auto create_fullscreen_pipeline(const Fullscreen &) -> CompiledPipeline;
    [[nodiscard]] auto get_or_create_fullscreen_vs(RenderContext &) -> u32;
} // namespace Pipeline
