#include "ArgumentParse.hxx"
#include "BindlessHeadless.hxx"
#include "BindlessSet.hxx"
#include "Buffer.hxx"
#include "Compiler.hxx"
#include "GlobalCommandContext.hxx"
#include "ImageOperations.hxx"
#include "Logger.hxx"
#include "PipelineCache.hxx"
#include "Pool.hxx"
#include "Reflection.hxx"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "3PP/PerlinNoise.hpp"

#define HAS_RENDERDOC
#ifdef HAS_RENDERDOC
#include "3PP/renderdoc_app.h"
#endif


#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

struct GpuPushConstants {
    const DeviceAddress ubo;
    const DeviceAddress lights;
    const DeviceAddress flags;
    const DeviceAddress prefix;
    const DeviceAddress group_sums;
    const DeviceAddress group_offsets;
    const DeviceAddress compact;
    const DeviceAddress indirect_cube;
    const u32 image_index;
    const u32 light_count;
    const u32 group_count;
};

struct RenderingPushConstants {
    const DeviceAddress ubo;
    const DeviceAddress compact;
    const DeviceAddress cube_vertices;
    const DeviceAddress cube_indices;
    const DeviceAddress indirect_cube;
};

struct FrustumPlane {
    glm::vec4 plane; // xyz = normal, w = distance
};

auto extract_frustum_planes = [](const glm::mat4 &inv_proj) -> std::array<FrustumPlane, 6> {
    // 1. Correct NDC Corners for ZO (0 to 1)
    constexpr std::array<glm::vec4, 8> ndc_corners = {
        glm::vec4{-1, -1, 0, 1}, {1, -1, 0, 1}, {-1, 1, 0, 1}, {1, 1, 0, 1}, // Near (0-3)
        glm::vec4{-1, -1, 1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}, {1, 1, 1, 1}  // Far  (4-7)
    };

    glm::vec3 v[8];
    for (int i = 0; i < 8; ++i) {
        glm::vec4 p = inv_proj * ndc_corners[i];
        v[i] = glm::vec3(p) / p.w;
    }

    auto compute_plane = [](glm::vec3 a, glm::vec3 b, glm::vec3 c) {
        // This order ensures the normal points INSIDE the frustum
        glm::vec3 normal = glm::normalize(glm::cross(c - a, b - a));
        return glm::vec4(normal, -glm::dot(normal, a));
    };

    std::array<FrustumPlane, 6> planes;
    planes[0].plane = compute_plane(v[0], v[2], v[4]); // Left
    planes[1].plane = compute_plane(v[1], v[5], v[3]); // Right
    planes[2].plane = compute_plane(v[0], v[4], v[1]); // Bottom
    planes[3].plane = compute_plane(v[2], v[3], v[6]); // Top
    planes[4].plane = compute_plane(v[0], v[1], v[2]); // Near
    planes[5].plane = compute_plane(v[4], v[6], v[5]); // Far

    return planes;
};

struct FrameUBO {
    glm::mat4 view;
    glm::mat4 projection;
    glm::mat4 view_projection;
    glm::mat4 inv_projection;
    glm::vec4 camera_position;
    std::array<FrustumPlane, 6> frustum_planes; // left, right, bottom, top, near, far
    float time;
    float delta_time;
    float _padding[2];
};


auto generate_perlin(auto w, auto h) -> std::vector<std::uint8_t> {
    std::vector<std::uint8_t> data;
    data.resize(w * h);
    const auto seed = static_cast<u32>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const siv::PerlinNoise pn{seed};

    auto z_offset = 0.0;
    for (auto y = 0; y < h; ++y) {
        const auto row_z = z_offset + static_cast<double>(y) * 0.01;
        for (auto x = 0; x < w; ++x) {
            const auto nx = static_cast<double>(x) / static_cast<double>(w);
            auto ny = static_cast<double>(y) / static_cast<double>(h);
            auto value = pn.noise3D(nx * 8.0, ny * 8.0, row_z);
            value = (value + 1.0) / 2.0;
            data[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)] =
                    static_cast<std::uint8_t>(value * 255.0);
        }
        z_offset += 0.0001;
    }

    return data;
}

static VkBool32 debug_callback(const VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                               VkDebugUtilsMessageTypeFlagsEXT type,
                               const VkDebugUtilsMessengerCallbackDataEXT *callback_data, void *) {
    if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        error("Validation layer: {}", callback_data->pMessage);
    }

    if (type & VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT) {
        info("What is this?: {}", callback_data->pMessage);
    }
    return VK_FALSE;
}

auto create_compute_pipeline(VkDevice device, PipelineCache &cache, VkDescriptorSetLayout layout,
                             const std::vector<u32> &code, const std::string_view entry_name)
        -> std::pair<VkPipeline, VkPipelineLayout> {
    VkShaderModule compute_shader{};
    VkShaderModuleCreateInfo create_info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                         .pNext = nullptr,
                                         .flags = 0,
                                         .codeSize = code.size() * sizeof(u32),
                                         .pCode = code.data()};
    vk_check(vkCreateShaderModule(device, &create_info, nullptr, &compute_shader));

    VkPushConstantRange push_constant_range{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(GpuPushConstants),
    };

    VkPipelineLayout pi_layout{};
    VkPipelineLayoutCreateInfo plci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant_range,
    };
    vk_check(vkCreatePipelineLayout(device, &plci, nullptr, &pi_layout));

    VkComputePipelineCreateInfo cpci{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                     .pNext = nullptr,
                                     .flags = 0,
                                     .stage =
                                             VkPipelineShaderStageCreateInfo{
                                                     .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                     .pNext = nullptr,
                                                     .flags = 0,
                                                     .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                                     .module = compute_shader,
                                                     .pName = entry_name.data(),
                                                     .pSpecializationInfo = nullptr,
                                             },
                                     .layout = pi_layout,
                                     .basePipelineHandle = VK_NULL_HANDLE,
                                     .basePipelineIndex = -1};
    VkPipeline pipeline{};
    vk_check(vkCreateComputePipelines(device, cache, 1, &cpci, nullptr, &pipeline));

    vkDestroyShaderModule(device, compute_shader, nullptr);
    return std::make_pair(pipeline, pi_layout);
}

auto create_graphics_pipeline(VkDevice device, PipelineCache &cache, VkDescriptorSetLayout layout,
                              const std::vector<u32> &vert_code, const std::vector<u32> &frag_code,
                              const std::string_view vert_entry, const std::string_view frag_entry,
                              VkFormat color_format)
        -> std::pair<VkPipeline, VkPipelineLayout> {
    VkShaderModule vert_shader{};
    VkShaderModuleCreateInfo vert_create_info{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = vert_code.size() * sizeof(u32),
            .pCode = vert_code.data()
    };
    vk_check(vkCreateShaderModule(device, &vert_create_info, nullptr, &vert_shader));

    VkShaderModule frag_shader{};
    VkShaderModuleCreateInfo frag_create_info{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = frag_code.size() * sizeof(u32),
            .pCode = frag_code.data()
    };
    vk_check(vkCreateShaderModule(device, &frag_create_info, nullptr, &frag_shader));

    VkPushConstantRange push_constant_range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(RenderingPushConstants),
    };

    VkPipelineLayout pipeline_layout{};
    VkPipelineLayoutCreateInfo plci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant_range,
    };
    vk_check(vkCreatePipelineLayout(device, &plci, nullptr, &pipeline_layout));

    std::array shader_stages{
            VkPipelineShaderStageCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .stage = VK_SHADER_STAGE_VERTEX_BIT,
                    .module = vert_shader,
                    .pName = vert_entry.data(),
                    .pSpecializationInfo = nullptr,
            },
            VkPipelineShaderStageCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                    .module = frag_shader,
                    .pName = frag_entry.data(),
                    .pSpecializationInfo = nullptr,
            }
    };

    VkPipelineVertexInputStateCreateInfo vertex_input{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .vertexBindingDescriptionCount = 0,
            .pVertexBindingDescriptions = nullptr,
            .vertexAttributeDescriptionCount = 0,
            .pVertexAttributeDescriptions = nullptr,
    };

    VkPipelineInputAssemblyStateCreateInfo input_assembly{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE,
    };

    VkPipelineViewportStateCreateInfo viewport_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .viewportCount = 1,
            .pViewports = nullptr, // dynamic
            .scissorCount = 1,
            .pScissors = nullptr, // dynamic
    };

    VkPipelineRasterizationStateCreateInfo rasterization{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_NONE,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .depthBiasConstantFactor = 0.0f,
            .depthBiasClamp = 0.0f,
            .depthBiasSlopeFactor = 0.0f,
            .lineWidth = 1.0f,
    };

    VkPipelineMultisampleStateCreateInfo multisample{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
            .minSampleShading = 1.0f,
            .pSampleMask = nullptr,
            .alphaToCoverageEnable = VK_FALSE,
            .alphaToOneEnable = VK_FALSE,
    };

    VkPipelineDepthStencilStateCreateInfo depth_stencil{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = VK_COMPARE_OP_GREATER,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE,
            .front = {},
            .back = {},
            .minDepthBounds = 1.0f,
            .maxDepthBounds = 0.0f,
    };

    VkPipelineColorBlendAttachmentState color_blend_attachment{
            .blendEnable = VK_TRUE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    VkPipelineColorBlendStateCreateInfo color_blend{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .logicOpEnable = VK_FALSE,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = 1,
            .pAttachments = &color_blend_attachment,
            .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f},
    };

    std::array dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
            .pDynamicStates = dynamic_states.data(),
    };

    VkPipelineRenderingCreateInfo rendering_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .pNext = nullptr,
            .viewMask = 0,
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &color_format,
            .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT,
            .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
    };

    VkGraphicsPipelineCreateInfo pipeline_info{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = &rendering_info,
            .flags = 0,
            .stageCount = static_cast<u32>(shader_stages.size()),
            .pStages = shader_stages.data(),
            .pVertexInputState = &vertex_input,
            .pInputAssemblyState = &input_assembly,
            .pTessellationState = nullptr,
            .pViewportState = &viewport_state,
            .pRasterizationState = &rasterization,
            .pMultisampleState = &multisample,
            .pDepthStencilState = &depth_stencil,
            .pColorBlendState = &color_blend,
            .pDynamicState = &dynamic_state,
            .layout = pipeline_layout,
            .renderPass = VK_NULL_HANDLE,
            .subpass = 0,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
    };

    VkPipeline pipeline{};
    vk_check(vkCreateGraphicsPipelines(device, cache, 1, &pipeline_info, nullptr, &pipeline));

    vkDestroyShaderModule(device, vert_shader, nullptr);
    vkDestroyShaderModule(device, frag_shader, nullptr);

    return std::make_pair(pipeline, pipeline_layout);
}

auto execute(int argc, char** argv) -> int {
    #ifdef HAS_RENDERDOC
    RENDERDOC_API_1_6_0 *rdoc_api = nullptr;
    if (HMODULE mod = GetModuleHandleA("renderdoc.dll")) {
        auto RENDERDOC_GetAPI = reinterpret_cast<pRENDERDOC_GetAPI>(GetProcAddress(mod, "RENDERDOC_GetAPI"));
        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0, reinterpret_cast<void **>(&rdoc_api));
        (void) ret;
    }
#endif
    auto opts = parse_cli(argc, argv);

    auto width = opts.width;
    auto height = opts.height;
    const auto aspect = static_cast<float>(width) / static_cast<float>(height);


    auto compiler = CompilerSession{};

    constexpr bool is_release = static_cast<bool>(IS_RELEASE);

    auto instance = create_instance_with_debug(debug_callback, is_release);
    auto could_choose = pick_physical_device(instance.instance);
    if (!could_choose) {
        return 1;
    }

    auto &&[physical_device, graphics_index, compute_index] = *could_choose;


    auto &&[device, graphics_queue, compute_queue] = create_device(physical_device, graphics_index, compute_index);

    auto cache_path = pipeline_cache_path(argc, argv);
    auto pipeline_cache = std::make_unique<PipelineCache>(device, opts.pipeline_cache_dir);

    auto command_context = create_global_cmd_context(device, graphics_queue, graphics_index);

    auto flags_code = compiler.compile_compute_from_file("shaders/light_cull_prefix_compact.slang", "LightFlagsCS");
    auto scan_local_code =
            compiler.compile_compute_from_file("shaders/light_cull_prefix_compact.slang", "LightScanLocalCS");
    auto scan_groups_code =
            compiler.compile_compute_from_file("shaders/light_cull_prefix_compact.slang", "LightScanGroupsCS");
    auto compact_code = compiler.compile_compute_from_file("shaders/light_cull_prefix_compact.slang", "LightCompactCS");
    auto setup_indirect_code = compiler.compile_compute_from_file("shaders/light_cull_prefix_compact.slang", "SetupIndirectCS");

    auto point_light_vert = compiler.compile_entry_from_file("shaders/point_light.slang", "main_vs");
    auto point_light_frag = compiler.compile_entry_from_file("shaders/point_light.slang", "heat_fs");

    auto allocator = create_allocator(instance.instance, physical_device, device);

    auto tl_compute = create_timeline(device, compute_queue, compute_index);
    auto tl_graphics = create_timeline(device, graphics_queue, graphics_index);

    BindlessCaps caps = query_bindless_caps(physical_device);
    BindlessSet bindless{};
    bindless.init(device, caps, 8u, 8u, 8u, 0u);

    bindless.grow_if_needed(300u, 40u, 32u, 8u);

    auto &&[flags_pipeline, flags_layout] =
            create_compute_pipeline(device, *pipeline_cache, bindless.layout, flags_code, "LightFlagsCS");
    auto &&[scan_local_pipeline, scan_local_layout] =
            create_compute_pipeline(device, *pipeline_cache, bindless.layout, scan_local_code, "LightScanLocalCS");
    auto &&[scan_groups_pipeline, scan_groups_layout] =
            create_compute_pipeline(device, *pipeline_cache, bindless.layout, scan_groups_code, "LightScanGroupsCS");
    auto &&[compact_pipeline, compact_layout] =
            create_compute_pipeline(device, *pipeline_cache, bindless.layout, compact_code, "LightCompactCS");
    auto &&[setup_indirect_pipeline, setup_indirect_layout] =
        create_compute_pipeline(device, *pipeline_cache, bindless.layout, setup_indirect_code, "SetupIndirectCS");
    auto &&[point_light_pipeline, point_light_layout] =
    create_graphics_pipeline(device, *pipeline_cache, bindless.layout,
                            point_light_vert, point_light_frag,
                            "main_vs", "heat_fs",
                            VK_FORMAT_R8G8B8A8_UNORM);

    DestructionContext ctx{
            .allocator = allocator,
            .bindless_set = &bindless,
    };

    std::array<DestructionContext::QueryPoolHandle, frames_in_flight> compute_query_pool{};
    std::array<DestructionContext::QueryPoolHandle, frames_in_flight> graphics_query_pool{};
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical_device, &props);
        const auto timestamp_period_ns = static_cast<double>(props.limits.timestampPeriod);

        for (u32 fi = 0; fi < frames_in_flight; ++fi) {
            VkQueryPoolCreateInfo qpci{.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                       .pNext = nullptr,
                                       .flags = 0,
                                       .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                       .queryCount = query_count,
                                       .pipelineStatistics = 0};

            VkQueryPool qpc = VK_NULL_HANDLE;
            vk_check(vkCreateQueryPool(device, &qpci, nullptr, &qpc));
            compute_query_pool[fi] = ctx.create_query_pool(QueryPoolState{
                    .pool = qpc, .query_count = query_count, .timestamp_period_ns = timestamp_period_ns});

            VkQueryPool qpg = VK_NULL_HANDLE;
            vk_check(vkCreateQueryPool(device, &qpci, nullptr, &qpg));
            graphics_query_pool[fi] = ctx.create_query_pool(QueryPoolState{
                    .pool = qpg, .query_count = query_count, .timestamp_period_ns = timestamp_period_ns});
        }
    }

    ctx.create_texture(create_offscreen_target(allocator, width, height, VK_FORMAT_R8G8B8A8_UNORM, "white-texture"));
    ctx.create_texture(create_offscreen_target(allocator, width, height, VK_FORMAT_R8G8B8A8_UNORM, "black-texture"));

    const auto noise = generate_perlin(2048, 2048);
    auto perlin_handle = ctx.create_texture(create_image_from_span_v2(
            allocator, command_context, 2048u, 2048u, VK_FORMAT_R8_UNORM, std::span{noise}, "perlin_noise"));

    auto offscreen_target_handle =
            ctx.create_texture(create_offscreen_target(allocator, width, height, VK_FORMAT_R8G8B8A8_UNORM, "offscreen"));

    auto offscreen_depth_target_handle =
        ctx.create_texture(create_depth_target(allocator, width, height, VK_FORMAT_D32_SFLOAT, "offscreen_depth"));

    ctx.create_sampler(
            VkSamplerCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .magFilter = VK_FILTER_LINEAR,
                    .minFilter = VK_FILTER_LINEAR,
                    .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                    .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    .mipLodBias = 0.0f,
                    .anisotropyEnable = VK_FALSE,
                    .maxAnisotropy = 1.0f,
                    .compareEnable = VK_FALSE,
                    .compareOp = VK_COMPARE_OP_ALWAYS,
                    .minLod = 0.0f,
                    .maxLod = VK_LOD_CLAMP_NONE,
                    .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
                    .unnormalizedCoordinates = VK_FALSE,
            },
            "linear_repeat");

    ctx.create_sampler(create_sampler(allocator,
                                                            VkSamplerCreateInfo{
                                                                    .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                                                    .magFilter = VK_FILTER_LINEAR,
                                                                    .minFilter = VK_FILTER_LINEAR,
                                                                    .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                                                    .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                                                    .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                                                    .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                                                    .mipLodBias = 0.0f,
                                                                    .anisotropyEnable = false,
                                                                    .compareEnable = false,
                                                                    .minLod = 0.0f,
                                                                    .maxLod = VK_LOD_CLAMP_NONE,
                                                                    .unnormalizedCoordinates = false,
                                                            },
                                                            "noise_sampler"));

    bindless.repopulate_if_needed(ctx.textures, ctx.samplers);

    std::array<FrameState, frames_in_flight> frames{};
    std::uint64_t i = 0;

    struct PointLight {
        std::array<float, 4> position_radius;
        std::array<float, 4> colour_intensity;
    };

    auto all_point_lights = std::vector<PointLight>(opts.light_count);
    auto all_point_lights_zero = std::vector<PointLight>(opts.light_count);
    auto light_count = static_cast<u32>(all_point_lights.size());
    constexpr u32 threads_per_group = 64u;
    auto group_count = (light_count + threads_per_group - 1u) / threads_per_group;

    constexpr auto world_size = 200.F;

    auto rng = std::default_random_engine{};
    auto distrib = std::uniform_real_distribution{-world_size, world_size};

    for (u32 idx = 0; idx < light_count; ++idx) {
        auto t = static_cast<float>(idx) / static_cast<float>(light_count);
        auto &[position_radius, colour_intensity] = all_point_lights[idx];

        position_radius = {distrib(rng), distrib(rng), distrib(rng), 5.0F};
        colour_intensity = {t, 1.0f - t, 0.5f, 1.0f};
    }

    auto point_light_handle = ctx.buffers.create(
            Buffer::from_slice<PointLight>(allocator,
                                           VkBufferCreateInfo{
                                                   .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                           },
                                           VmaAllocationCreateInfo{}, all_point_lights, "point_light")
                    .value());

    std::vector zeros_lights(light_count, 0u);
    std::vector zeros_groups(group_count, 0u);
    auto flags_handle =
            ctx.buffers.create(Buffer::from_slice<u32>(allocator,
                                                       VkBufferCreateInfo{
                                                               .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                       },
                                                       VmaAllocationCreateInfo{}, zeros_lights, "light_flags")
                                       .value());
    auto prefix_handle =
            ctx.buffers.create(Buffer::from_slice<u32>(allocator,
                                                       VkBufferCreateInfo{
                                                               .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                       },
                                                       VmaAllocationCreateInfo{}, zeros_lights, "light_prefix")
                                       .value());
    auto group_sums_handle =
            ctx.buffers.create(Buffer::from_slice<u32>(allocator,
                                                       VkBufferCreateInfo{
                                                               .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                       },
                                                       VmaAllocationCreateInfo{}, zeros_groups, "group_sums")
                                       .value());
    auto group_offsets_handle =
            ctx.buffers.create(Buffer::from_slice<u32>(allocator,
                                                       VkBufferCreateInfo{
                                                               .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                       },
                                                       VmaAllocationCreateInfo{}, zeros_groups, "group_offsets")
                                       .value());
    auto compact_lights_handle = ctx.buffers.create(
            Buffer::from_slice<PointLight>(
                    allocator,
                    VkBufferCreateInfo{
                            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    },
                    VmaAllocationCreateInfo{}, all_point_lights_zero, "compact_lights")
                    .value());

    VkDrawIndexedIndirectCommand indirect_cmd = {
        .indexCount = 36,
        .instanceCount = 0,
        .firstIndex = 0,
        .vertexOffset = 0,
        .firstInstance = 0,
    };

    auto indirect_buffer_handle = ctx.buffers.create(
        Buffer::from_slice<VkDrawIndexedIndirectCommand>(
            allocator,
            VkBufferCreateInfo{
                .usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            },
            VmaAllocationCreateInfo{},
            std::span{&indirect_cmd, 1},
            "light_draw_indirect"
        ).value()
    );

    struct Vert {
        glm::vec3 pos;
    };

    struct Cube {
        std::array<Vert, 8> verts;
    };

   constexpr Cube cube = {
        glm::vec3{-0.5f, -0.5f, -0.5f},
        glm::vec3{ 0.5f, -0.5f, -0.5f},
        glm::vec3{ 0.5f,  0.5f, -0.5f},
        glm::vec3{-0.5f,  0.5f, -0.5f},
        glm::vec3{-0.5f, -0.5f,  0.5f},
        glm::vec3{ 0.5f, -0.5f,  0.5f},
        glm::vec3{ 0.5f,  0.5f,  0.5f},
        glm::vec3{-0.5f,  0.5f,  0.5f},
    };

    constexpr std::array<u32, 36> cube_indices = {
        // back (-Z)
        2, 1, 0, 0, 3, 2,

        // front (+Z)
        4, 5, 6, 6, 7, 4,

        // bottom (-Y)
        0, 1, 5, 5, 4, 0,

        // top (+Y)
        3, 7, 6, 6, 2, 3,

        // left (-X)
        0, 4, 7, 7, 3, 0,

        // right (+X)
        1, 2, 6, 6, 5, 1,
    };

    auto cube_vertices_handle = ctx.buffers.create(
        Buffer::from_slice<Cube>(
            allocator,
            VkBufferCreateInfo{
                .usage =  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            },
            VmaAllocationCreateInfo{},
            std::span {&cube, 1},
            "light_draw_indirect"
        ).value()
    );

    auto cube_indices_handle = ctx.buffers.create(
    Buffer::from_slice<u32>(
        allocator,
        VkBufferCreateInfo{
            .usage =  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        },
        VmaAllocationCreateInfo{},
        std::span {cube_indices.data(), cube_indices.size()},
        "light_draw_indirect"
    ).value()
);

    glm::vec3 camera_pos = glm::vec3(0, 50, -100);
    glm::vec3 camera_target = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 camera_up = glm::vec3(0.0f, 1.0f, 0.0f);

    glm::mat4 view = glm::lookAt(camera_pos, camera_target, camera_up);
    glm::mat4 projection = glm::perspective(glm::radians(90.0f), // FOV
                                            aspect, // aspect ratio
                                            0.1f, // near plane
                                            1000.0f // far plane
    );

    glm::mat4 inv_proj = glm::inverse(projection);
    auto frustum_planes = extract_frustum_planes(inv_proj);

    FrameUBO ubo_data{
            .view = view,
            .projection = projection,
            .view_projection = projection * view,
            .inv_projection = inv_proj,
            .camera_position = glm::vec4(camera_pos, 1.0f),
            .frustum_planes = {frustum_planes[0], frustum_planes[1], frustum_planes[2], frustum_planes[3],
                               frustum_planes[4], frustum_planes[5]},
            .time = 0.0f,
            .delta_time = 0.0f,
    };

    // Create the UBO buffer
    auto frame_ubo_handle = ctx.buffers.create(
            Buffer::from_slice<FrameUBO>(allocator,
                                         VkBufferCreateInfo{
                                                 .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                         },
                                         VmaAllocationCreateInfo{}, std::span{&ubo_data, 1}, "frame_ubo")
                    .value());

    auto frame_ubo = ctx.device_address(frame_ubo_handle);

    auto light_addr = ctx.device_address(point_light_handle);
    auto flags_addr = ctx.device_address(flags_handle);
    auto prefix_addr = ctx.device_address(prefix_handle);
    auto group_sums_addr = ctx.device_address(group_sums_handle);
    auto group_offsets_addr = ctx.device_address(group_offsets_handle);
    auto compact_addr = ctx.device_address(compact_lights_handle);
    auto indirect_addr = ctx.device_address(indirect_buffer_handle);
    auto cube_vertices_addr = ctx.device_address(cube_vertices_handle);
    auto cube_indices_addr = ctx.device_address(cube_indices_handle);
    auto indirect_buf = ctx.buffers.get(indirect_buffer_handle)->buffer();

    auto stats = FrameStats{};
    FrameStats gpu_compute_ms{};
    FrameStats gpu_graphics_ms{};

    auto read_timestamp_ms = [&](const auto h) -> std::optional<double> {
        const auto *qs = ctx.query_pools.get(h);
        if (!qs)
            return std::nullopt;

        u64 stamps[2] = {};
        const auto r = vkGetQueryPoolResults(device, qs->pool, 0, 2, sizeof(stamps), stamps, sizeof(u64),
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (r != VK_SUCCESS)
            return std::nullopt;

        const u64 dt_ticks = stamps[1] - stamps[0];
        const double dt_ns = static_cast<double>(dt_ticks) * qs->timestamp_period_ns;
        constexpr auto ns_to_ms_factor = 1e-6;
        return dt_ns * ns_to_ms_factor;
    };

    for (i = 0; i < opts.iteration_count; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

#ifdef HAS_RENDERDOC
        if (rdoc_api)
            rdoc_api->StartFrameCapture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(instance.instance), NULL);
#endif
        bindless.repopulate_if_needed(ctx.textures, ctx.samplers);

        const auto frame_index = static_cast<u32>(i % frames_in_flight);
        auto &fs = frames[frame_index];

        if (fs.frame_done_value > 0) {
            VkSemaphoreWaitInfo wi{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                   .pNext = nullptr,
                                   .flags = 0,
                                   .semaphoreCount = 1,
                                   .pSemaphores = &tl_graphics.timeline,
                                   .pValues = &fs.frame_done_value};
            vk_check(vkWaitSemaphores(device, &wi, UINT64_MAX));

            if (auto ms = read_timestamp_ms(compute_query_pool[frame_index])) {
                gpu_compute_ms.add_sample(*ms);
            }
            if (auto ms = read_timestamp_ms(graphics_query_pool[frame_index])) {
                gpu_graphics_ms.add_sample(*ms);
            }
        }

        std::array<VkSemaphore, 0> no_wait_sems{};
        std::array<std::uint64_t, 0> no_wait_vals{};

        auto light_val = submit_stage(
                tl_compute, device,
                [&](VkCommandBuffer cmd) {
                    const auto *cqs = ctx.query_pools.get(compute_query_pool[frame_index]);
                    const auto& cqp = cqs->pool;

                    vkCmdResetQueryPool(cmd, cqp, 0, cqs->query_count);
                    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, cqp,
                                        static_cast<u32>(GpuStamp::Begin));
                    const GpuPushConstants pc{
                            .ubo = frame_ubo,
                            .lights = light_addr,
                            .flags = flags_addr,
                            .prefix = prefix_addr,
                            .group_sums = group_sums_addr,
                            .group_offsets = group_offsets_addr,
                            .compact = compact_addr,
                        .indirect_cube = indirect_addr,
                            .image_index = offscreen_target_handle.index(),
                            .light_count = light_count,
                            .group_count = group_count,
                    };

                    auto bind_and_dispatch = [&](VkPipeline pipeline, VkPipelineLayout layout, u32 groups_x) {
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &bindless.set, 0,
                                                nullptr);

                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

                        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GpuPushConstants), &pc);

                        vkCmdDispatch(cmd, groups_x, 1u, 1u);
                    };

                    VkMemoryBarrier mem_barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                                .pNext = nullptr,
                                                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                                .dstAccessMask =
                                                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};

                    bind_and_dispatch(flags_pipeline, flags_layout, group_count);
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem_barrier, 0, nullptr, 0,
                                         nullptr);

                    bind_and_dispatch(scan_local_pipeline, scan_local_layout, group_count);
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem_barrier, 0, nullptr, 0,
                                         nullptr);

                    bind_and_dispatch(scan_groups_pipeline, scan_groups_layout, group_count);
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem_barrier, 0, nullptr, 0,
                                         nullptr);

                    bind_and_dispatch(compact_pipeline, compact_layout, group_count);
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem_barrier, 0, nullptr, 0,
                                         nullptr);

                    // 2. Dispatch Setup: One thread to rule them all
                    bind_and_dispatch(setup_indirect_pipeline, setup_indirect_layout, 1u);

                    VkBufferMemoryBarrier indirect_barrier{
                        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                        .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT, // Essential for DrawIndirect
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .buffer = indirect_buf,
                        .offset = 0,
                        .size = VK_WHOLE_SIZE
                    };

                    vkCmdPipelineBarrier(cmd,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,      // Source: Your setup shader
                        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,       // Destination: The Indirect command processor
                        0, 0, nullptr, 1, &indirect_barrier, 0, nullptr);


                    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, cqp,
                                        static_cast<u32>(GpuStamp::End));
                },
                std::span{no_wait_sems}, std::span{no_wait_vals});

        fs.timeline_values[stage_index(Stage::LightCulling)] = light_val;

        std::array gbuffer_wait_sems{tl_compute.timeline};
        std::array gbuffer_wait_vals{fs.timeline_values[stage_index(Stage::LightCulling)]};

       auto gbuffer_val = submit_stage(
        tl_graphics, device,
        [&](VkCommandBuffer cmd) {
            const auto *gqs = ctx.query_pools.get(graphics_query_pool[frame_index]);
            auto&& [offscreen, depth] = ctx.textures.get_multiple(offscreen_target_handle, offscreen_depth_target_handle);
            const VkQueryPool& gqp = gqs->pool;

            vkCmdResetQueryPool(cmd, gqp, 0, gqs->query_count);
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, gqp, 0);

            // Only transition on first use
            if (!offscreen->initialized) {
                VkImageMemoryBarrier color_barrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = 0,
                    .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = offscreen->image,
                    .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                    }
                };

                vkCmdPipelineBarrier(
                    cmd,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    0,
                    0, nullptr,
                    0, nullptr,
                    1, &color_barrier
                );

                offscreen->initialized = true;
            }

            if (!depth->initialized) {
                VkImageMemoryBarrier depth_barrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = 0,
                    .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = depth->image,
                    .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                    }
                };

                vkCmdPipelineBarrier(
                    cmd,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                    0,
                    0, nullptr,
                    0, nullptr,
                    1, &depth_barrier
                );

                depth->initialized = true;
            }

            VkRenderingAttachmentInfo color_attachment{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .imageView = offscreen->sampled_view,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue = {.color = {0.0f, 0.0f, 0.0f, 1.0f}},
            };

            VkRenderingAttachmentInfo depth_attachment{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .imageView = depth->sampled_view,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .clearValue = {.depthStencil = {0.0f, 0}},
            };

            VkRenderingInfo rendering_info{
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .renderArea = {.offset = {0, 0}, .extent = {width, height}},
                .layerCount = 1,
                .colorAttachmentCount = 1,
                .pColorAttachments = &color_attachment,
                .pDepthAttachment = &depth_attachment,
            };

            vkCmdBeginRendering(cmd, &rendering_info);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, point_light_pipeline);

            const RenderingPushConstants pc {
                .ubo = frame_ubo,
                .compact = compact_addr,
                .cube_vertices = cube_vertices_addr,
                .cube_indices = cube_indices_addr,
                .indirect_cube = indirect_addr,
            };

            const auto w = static_cast<float>(width);
            const auto h = static_cast<float>(height);
            VkViewport vp {
                .x = 0,
                .y = h,
                .width = w,
                .height = -h,
                .minDepth = 1.0f,
                .maxDepth = 0.0f,
            };

            VkRect2D sc {
                .offset = {0, 0},
                .extent = {width, height}
            };

            vkCmdSetViewport(cmd, 0, 1, &vp);
            vkCmdSetScissor(cmd, 0, 1, &sc);
            vkCmdPushConstants(cmd, point_light_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
            vkCmdDrawIndirect(cmd, indirect_buf, 0, 1, sizeof(VkDrawIndexedIndirectCommand));

            vkCmdEndRendering(cmd);

            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, gqp, 1);
        },
        std::span{gbuffer_wait_sems}, std::span{gbuffer_wait_vals});

        fs.timeline_values[stage_index(Stage::GBuffer)] = gbuffer_val;
        fs.frame_done_value = gbuffer_val;

        throttle(tl_compute, device);
        throttle(tl_graphics, device);

        const auto completed = std::min(tl_compute.completed, tl_graphics.completed);
        ctx.destroy_queue.retire(completed);
#ifdef HAS_RENDERDOC
        if (rdoc_api)
            rdoc_api->EndFrameCapture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(instance.instance), NULL);
#endif
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(frame_end - start_time).count();
        stats.add_sample(ms);
    }

    info("Light count {}", opts.light_count);
    info("frames: {}", stats.samples.size());
    info("mean:   {:.3f} ms", stats.mean);
    info("median: {:.3f} ms", stats.median());
    info("stddev: {:.3f} ms", stats.stddev_sample());
    info("quartiles: {}", stats.quartiles());
    info("Total: {:.3f} s", stats.total() / 1000.0F);

    info("GPU compute mean:   {:.3f} ms", gpu_compute_ms.avg());
    info("GPU compute p95:    {:.3f} ms", gpu_compute_ms.p95());
    info("GPU graphics mean:  {:.3f} ms", gpu_graphics_ms.avg());
    info("GPU graphics p95:   {:.3f} ms", gpu_graphics_ms.p95());

        const auto&& [oth, ph] = ctx.textures.get_multiple(offscreen_target_handle, perlin_handle);
       image_operations::write_to_disk(oth, allocator, "output.bmp");
        image_operations::write_to_disk(ph, allocator, "perlin.bmp");

    pipeline_cache.reset();
    ctx.clear_all();

    ctx.destroy_queue.retire(UINT64_MAX);

    destruction::pipeline(device, flags_pipeline, flags_layout);
    destruction::pipeline(device, scan_local_pipeline, scan_local_layout);
    destruction::pipeline(device, scan_groups_pipeline, scan_groups_layout);
    destruction::pipeline(device, compact_pipeline, compact_layout);
    destruction::pipeline(device, point_light_pipeline, point_light_layout);
    destruction::pipeline(device, setup_indirect_pipeline, setup_indirect_layout);
    destruction::global_command_context(command_context);
    destruction::bindless_set(device, bindless);
    destruction::timeline_compute(device, tl_graphics);
    destruction::timeline_compute(device, tl_compute);
    destruction::allocator(allocator);
        vkDeviceWaitIdle(device);
    destruction::device(device);
    destruction::instance(instance);

    return 0;
}


auto main(int argc , char ** argv) -> int {
    execute(argc, argv);

    info("Bindless headless setup and teardown completed successfully.");
    return 0;
}
