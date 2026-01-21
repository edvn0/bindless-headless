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
#include "ResizeableGraph.hxx"

#include <GLFW/glfw3.h>
#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <ranges>
#include <thread>
#include <tracy/Tracy.hpp>

#include "3PP/PerlinNoise.hpp"
#include "Profiler.hxx"


#ifdef HAS_RENDERDOC
#include "3PP/renderdoc_app.h"
#endif


#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include "Swapchain.hxx"

struct GpuPushConstants {
    const DeviceAddress ubo;
    const DeviceAddress lights;
    const DeviceAddress flags;
    const DeviceAddress prefix;
    const DeviceAddress group_sums;
    const DeviceAddress group_offsets;
    const DeviceAddress compact;
    const DeviceAddress indirect_point_light;
    const DeviceAddress indirect_meshlet;
    const u32 image_index;
    const u32 light_count;
    const u32 group_count;
};

struct PredepthPushConstants {
    const DeviceAddress ubo;
    const DeviceAddress cube_buffer;
    const DeviceAddress indirect_meshlet;
};

struct RenderingPushConstants {
    const DeviceAddress ubo;
    const DeviceAddress cubes;
    const DeviceAddress indirect_meshlet;
};

struct TonemapPushConstants {
    float exposure;
    const u32 image_index;
    const u32 sampler_index;
};

struct FrustumPlane {
    glm::vec4 plane; // xyz = normal, w = distance
};

auto fill_zeros(VkCommandBuffer cmd, auto &buffers_ctx, auto &&...buffer_handles) {
    (vkCmdFillBuffer(cmd, buffers_ctx.get(buffer_handles)->buffer(), 0, VK_WHOLE_SIZE, 0), ...);
}

auto extract_frustum_planes = [](const glm::mat4 &inv_proj) -> std::array<FrustumPlane, 6> {
    // 1. Correct NDC Corners for ZO (0 to 1)
    constexpr std::array<glm::vec4, 8> ndc_corners = {
            glm::vec4{-1, -1, 0, 1}, {1, -1, 0, 1}, {-1, 1, 0, 1}, {1, 1, 0, 1}, // Near (0-3)
            glm::vec4{-1, -1, 1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}, {1, 1, 1, 1} // Far  (4-7)
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

auto create_compute_pipeline(VkDevice device, PipelineCache &cache, VkDescriptorSetLayout layout,
                             const std::vector<u32> &code, const std::string_view entry_name) -> CompiledPipeline {
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
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, pi_layout, entry_name);

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
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, entry_name);


    vkDestroyShaderModule(device, compute_shader, nullptr);
    return {pipeline, pi_layout};
}

auto create_predepth_pipeline(VkDevice device, VkPipelineCache cache, VkDescriptorSetLayout bindless_layout,
                              const std::vector<uint32_t> &task_code, const std::vector<uint32_t> &mesh_spirv,
                              const std::vector<uint32_t> &frag_code, VkFormat depth_format) -> CompiledPipeline {
    VkShaderModule task_module{};
    VkShaderModuleCreateInfo create_info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                         .pNext = nullptr,
                                         .flags = 0,
                                         .codeSize = task_code.size() * sizeof(u32),
                                         .pCode = task_code.data()};
    vk_check(vkCreateShaderModule(device, &create_info, nullptr, &task_module));

    VkShaderModule mesh_module{};
    create_info = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                   .pNext = nullptr,
                   .flags = 0,
                   .codeSize = mesh_spirv.size() * sizeof(u32),
                   .pCode = mesh_spirv.data()};
    vk_check(vkCreateShaderModule(device, &create_info, nullptr, &mesh_module));

    VkShaderModule frag_module{};
    create_info = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                   .pNext = nullptr,
                   .flags = 0,
                   .codeSize = frag_code.size() * sizeof(u32),
                   .pCode = frag_code.data()};
    vk_check(vkCreateShaderModule(device, &create_info, nullptr, &frag_module));

    std::array stages = {
            VkPipelineShaderStageCreateInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                            .stage = VK_SHADER_STAGE_TASK_BIT_EXT,
                                            .module = task_module,
                                            .pName = "main_ts"},
            VkPipelineShaderStageCreateInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                            .stage = VK_SHADER_STAGE_MESH_BIT_EXT,
                                            .module = mesh_module,
                                            .pName = "main_ms"},
            VkPipelineShaderStageCreateInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                                            .module = frag_module,
                                            .pName = "main_fs"},
    };

    // 2. Pipeline Layout (Inherit bindless + push constants)
    VkPushConstantRange push_range{.stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT |
                                                 VK_SHADER_STAGE_FRAGMENT_BIT,
                                   .offset = 0,
                                   .size = sizeof(PredepthPushConstants)};

    VkPipelineLayoutCreateInfo layout_ci{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                         .setLayoutCount = 1,
                                         .pSetLayouts = &bindless_layout,
                                         .pushConstantRangeCount = 1,
                                         .pPushConstantRanges = &push_range};
    VkPipelineLayout layout;
    vkCreatePipelineLayout(device, &layout_ci, nullptr, &layout);
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, layout, "predepth");


    // 3. Specialized Depth State
    VkPipelineDepthStencilStateCreateInfo ds{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL, // Reverse-Z: Near is 1.0, Far is 0.0
            .minDepthBounds = 0.0f,
            .maxDepthBounds = 1.0f,
    };

    // 4. No Color Attachments (The secret to Pre-Depth speed)
    VkPipelineColorBlendStateCreateInfo cb{.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                           .attachmentCount = 0,
                                           .pAttachments = nullptr};

    // 5. Rasterization (Ensure Back-Face Culling is ON)
    VkPipelineRasterizationStateCreateInfo rs{.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                              .cullMode = VK_CULL_MODE_BACK_BIT,
                                              .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
                                              .lineWidth = 1.0f};

    // 6. Dynamic Rendering Info
    VkPipelineRenderingCreateInfo rendering_info{.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                                                 .colorAttachmentCount = 0,
                                                 .depthAttachmentFormat = depth_format};

    // Viewport/Scissor setup (Standard)
    VkPipelineViewportStateCreateInfo vp{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineMultisampleStateCreateInfo ms{.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    std::array dynamic_states = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy{.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                        .dynamicStateCount = 2,
                                        .pDynamicStates = dynamic_states.data()};

    VkGraphicsPipelineCreateInfo ci{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = &rendering_info,
            .stageCount = static_cast<uint32_t>(stages.size()),
            .pStages = stages.data(),
            .pViewportState = &vp,
            .pRasterizationState = &rs,
            .pMultisampleState = &ms,
            .pDepthStencilState = &ds,
            .pColorBlendState = &cb,
            .pDynamicState = &dy,
            .layout = layout,
    };

    VkPipeline pipeline;
    vkCreateGraphicsPipelines(device, cache, 1, &ci, nullptr, &pipeline);
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, "predepth");


    // Cleanup local modules
    vkDestroyShaderModule(device, task_module, nullptr);
    vkDestroyShaderModule(device, mesh_module, nullptr);
    vkDestroyShaderModule(device, frag_module, nullptr);

    return {pipeline, layout};
}

auto create_mesh_pipeline(VkDevice device, PipelineCache &cache, VkDescriptorSetLayout layout,
                          const std::vector<u32> &mesh_code, const std::vector<u32> &task_code,
                          const std::vector<u32> &frag_code, VkFormat color_format) -> CompiledPipeline {
    VkShaderModule mesh_module{};
    VkShaderModuleCreateInfo mesh_create_info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                              .pNext = nullptr,
                                              .flags = 0,
                                              .codeSize = mesh_code.size() * sizeof(u32),
                                              .pCode = mesh_code.data()};
    vk_check(vkCreateShaderModule(device, &mesh_create_info, nullptr, &mesh_module));

    VkShaderModule task_module{};
    VkShaderModuleCreateInfo task_create_info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                              .pNext = nullptr,
                                              .flags = 0,
                                              .codeSize = task_code.size() * sizeof(u32),
                                              .pCode = task_code.data()};
    vk_check(vkCreateShaderModule(device, &task_create_info, nullptr, &task_module));

    VkShaderModule frag_module{};
    VkShaderModuleCreateInfo frag_create_info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                              .pNext = nullptr,
                                              .flags = 0,
                                              .codeSize = frag_code.size() * sizeof(u32),
                                              .pCode = frag_code.data()};
    vk_check(vkCreateShaderModule(device, &frag_create_info, nullptr, &frag_module));

    VkPushConstantRange push_constant_range{
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
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
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, pipeline_layout, "mesh_primary");


    std::vector<VkPipelineShaderStageCreateInfo> stages = {
            {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
             .stage = VK_SHADER_STAGE_TASK_BIT_EXT,
             .module = task_module,
             .pName = "main_ts"},
            {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
             .stage = VK_SHADER_STAGE_MESH_BIT_EXT,
             .module = mesh_module,
             .pName = "main_ms"},
            {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
             .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
             .module = frag_module,
             .pName = "main_fs"}};

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
            .depthWriteEnable = VK_FALSE,
            .depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE,
            .front = {},
            .back = {},
            .minDepthBounds = 1.0f,
            .maxDepthBounds = 0.0f,
    };

    VkPipelineColorBlendAttachmentState color_blend_attachment{
            .blendEnable = VK_FALSE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                              VK_COLOR_COMPONENT_A_BIT,
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
            .stageCount = static_cast<u32>(stages.size()),
            .pStages = stages.data(),
            .pVertexInputState = nullptr,
            .pInputAssemblyState = nullptr,
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
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, "mesh_primary");


    vkDestroyShaderModule(device, mesh_module, nullptr);
    vkDestroyShaderModule(device, task_module, nullptr);
    vkDestroyShaderModule(device, frag_module, nullptr);

    return CompiledPipeline{pipeline, pipeline_layout};
}

auto create_tonemap_pipeline(VkDevice device, PipelineCache &cache, VkDescriptorSetLayout layout,
                             const std::vector<u32> &vert_code, const std::vector<u32> &frag_code,
                             const std::string_view vert_entry, const std::string_view frag_entry,
                             VkFormat color_format) -> CompiledPipeline {
    VkShaderModule vert_shader{};
    VkShaderModuleCreateInfo vert_create_info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                              .pNext = nullptr,
                                              .flags = 0,
                                              .codeSize = vert_code.size() * sizeof(u32),
                                              .pCode = vert_code.data()};
    vk_check(vkCreateShaderModule(device, &vert_create_info, nullptr, &vert_shader));

    VkShaderModule frag_shader{};
    VkShaderModuleCreateInfo frag_create_info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                              .pNext = nullptr,
                                              .flags = 0,
                                              .codeSize = frag_code.size() * sizeof(u32),
                                              .pCode = frag_code.data()};
    vk_check(vkCreateShaderModule(device, &frag_create_info, nullptr, &frag_shader));

    VkPushConstantRange push_constant_range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(TonemapPushConstants),
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

    std::array shader_stages{VkPipelineShaderStageCreateInfo{
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
                             }};

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
            .depthTestEnable = VK_FALSE,
            .depthWriteEnable = VK_FALSE,
            .depthCompareOp = VK_COMPARE_OP_ALWAYS,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE,
            .front = {},
            .back = {},
            .minDepthBounds = 1.0f,
            .maxDepthBounds = 0.0f,
    };

    VkPipelineColorBlendAttachmentState color_blend_attachment{
            .blendEnable = VK_FALSE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                              VK_COLOR_COMPONENT_A_BIT,
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
            .depthAttachmentFormat = VK_FORMAT_UNDEFINED,
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

    return CompiledPipeline{pipeline, pipeline_layout};
}

static MaybeNoOp<PFN_vkCmdDrawMeshTasksIndirectEXT> draw_mesh{};


auto execute(int argc, char **argv) -> int {
    if (auto init = glfwInit(); init != GLFW_TRUE) {
        error("Could not initialize GLFW");
        return 1;
    }

#ifdef HAS_RENDERDOC
    RENDERDOC_API_1_6_0 *rdoc_api = nullptr;
    if (HMODULE mod = GetModuleHandleA("renderdoc.dll")) {
        auto RENDERDOC_GetAPI = reinterpret_cast<pRENDERDOC_GetAPI>(GetProcAddress(mod, "RENDERDOC_GetAPI"));
        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0, reinterpret_cast<void **>(&rdoc_api));
        (void) ret;
    }
#endif
    auto opts = parse_cli(argc, argv);

    auto compiler = CompilerSession{};

    constexpr bool is_release = static_cast<bool>(IS_RELEASE);

    uint32_t count{};
    const char **extensions_raw = glfwGetRequiredInstanceExtensions(&count);
    std::vector<std::string_view> extensions(extensions_raw, extensions_raw + count);

    auto instance = create_instance_with_debug(debug_callback, extensions, is_release);
    auto could_choose = pick_physical_device(instance.instance);
    if (!could_choose) {
        return 1;
    }

    if (auto name = vkGetInstanceProcAddr(instance.instance, "vkCmdDrawMeshTasksIndirectEXT");
        name && draw_mesh.empty()) {
        draw_mesh = reinterpret_cast<PFN_vkCmdDrawMeshTasksIndirectEXT>(name);
    }

    auto &&[physical_device, graphics_index, compute_index] = *could_choose;
    auto &&[device, graphics_queue, compute_queue] = create_device(physical_device, graphics_index, compute_index);

    TracyGpuContext tracy_graphics{};
    TracyGpuContext tracy_compute{};
    tracy_graphics.init_calibrated(instance.instance, physical_device, device, graphics_queue, graphics_index,
                                   "Graphics Queue");
    tracy_compute.init_calibrated(instance.instance, physical_device, device, compute_queue, compute_index,
                                  "Compute Queue");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    auto window =
            glfwCreateWindow(static_cast<i32>(opts.width), static_cast<i32>(opts.height), "Bindless", nullptr, nullptr);
    if (!window) {
        error("Could not create window");
        return 1;
    }

    VkSurfaceKHR surface{};
    vk_check(glfwCreateWindowSurface(instance.instance, window, nullptr, &surface));

    auto maybe_swapchain = Swapchain::create(SwapchainCreateInfo{.physical_device = physical_device,
                                                                 .device = device,
                                                                 .surface = surface,
                                                                 .graphics_family = graphics_index,
                                                                 .extent = VkExtent2D{opts.width, opts.height},
                                                                 .vsync = opts.vsync});
    if (!maybe_swapchain) {
        return 1;
    }

    auto swapchain = std::move(maybe_swapchain.value());


    auto cache_path = pipeline_cache_path(argc, argv);
    auto pipeline_cache = std::make_unique<PipelineCache>(device, opts.pipeline_cache_dir);

    auto command_context = create_global_cmd_context(device, graphics_queue, graphics_index);

    std::array<const std::string_view, 5> names = {"LightFlagsCS", "LightScanLocalCS", "LightScanGroupsCS",
                                                   "LightCompactCS", "SetupIndirectCS"};
    std::array<ReflectionData, 5> reflection_data = {};
    auto culling_code = compiler.compile_from_file("shaders/light_cull_prefix_compact.slang", std::span(names),
                                                   std::span(reflection_data));

    std::array<const std::string_view, 3> point_light_names = {"main_ms", "main_ts", "main_fs"};
    std::array<ReflectionData, 3> point_light_reflection = {};
    auto point_light_code = compiler.compile_from_file("shaders/point_light.slang", std::span(point_light_names),
                                                       std::span(point_light_reflection));

    std::array<const std::string_view, 3> predepth_names{"main_ts", "main_ms", "main_fs"};
    std::array<ReflectionData, 3> predepth_reflection{};
    auto predepth_code = compiler.compile_from_file("shaders/predepth.slang", std::span(predepth_names),
                                                    std::span(predepth_reflection));

    std::array<const std::string_view, 2> tonemap_names{"vs_main", "fs_main"};
    std::array<ReflectionData, 2> tonemap_reflection{};
    auto tonemap_code = compiler.compile_from_file("shaders/tonemap.slang", std::span(tonemap_names),
                                                   std::span(tonemap_reflection));

    auto allocator = create_allocator(instance.instance, physical_device, device);

    auto tl_compute = create_compute_timeline(device, compute_queue, compute_index);
    auto tl_graphics = create_graphics_timeline(device, graphics_queue, graphics_index);

    BindlessCaps caps = query_bindless_caps(physical_device);
    BindlessSet bindless{};
    bindless.init(device, caps, 8u, 8u, 8u, 0u);

    bindless.grow_if_needed(300u, 40u, 32u, 8u);

    auto &&[flags, scan_local, scan_groups, compact, setup_indirect] = create_compute_pipelines(
            device, *pipeline_cache, bindless.layout, std::span(culling_code), std::span(names));

    auto point_light_pipeline =
            create_mesh_pipeline(device, *pipeline_cache, bindless.layout, point_light_code.at(0),
                                 point_light_code.at(1), point_light_code.at(2), VK_FORMAT_R32G32B32A32_SFLOAT);
    auto predepth_pipeline = create_predepth_pipeline(device, *pipeline_cache, bindless.layout, predepth_code.at(0),
                                                      predepth_code.at(1), predepth_code.at(2), VK_FORMAT_D32_SFLOAT);

    auto tonemap_pipeline = create_tonemap_pipeline(device, *pipeline_cache, bindless.layout, tonemap_code.at(0),
                                                    tonemap_code.at(1), "vs_main", "fs_main", VK_FORMAT_R8G8B8A8_UNORM);

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

    ctx.create_texture(
            create_offscreen_target(allocator, opts.width, opts.height, VK_FORMAT_R8G8B8A8_UNORM, "white-texture"));
    ctx.create_texture(
            create_offscreen_target(allocator, opts.width, opts.height, VK_FORMAT_R8G8B8A8_UNORM, "black-texture"));

    const auto noise = generate_perlin(2048, 2048);
    auto perlin_handle = ctx.create_texture(create_image_from_span_v2(
            allocator, command_context, 2048u, 2048u, VK_FORMAT_R8_UNORM, std::span{noise}, "perlin_noise"));

    auto offscreen_target_handle = ctx.create_texture(
            create_offscreen_target(allocator, opts.width, opts.height, VK_FORMAT_R32G32B32A32_SFLOAT, "offscreen"));

    auto tonemapped_target_handle = ctx.create_texture(
            create_offscreen_target(allocator, opts.width, opts.height, VK_FORMAT_R8G8B8A8_UNORM, "tonemapped"));

    auto offscreen_depth_target_handle = ctx.create_texture(
            create_depth_target(allocator, opts.width, opts.height, VK_FORMAT_D32_SFLOAT, "offscreen_depth"));

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

    auto rng = std::default_random_engine{
            static_cast<u32>(std::chrono::high_resolution_clock::now().time_since_epoch().count())};
    auto distrib = std::uniform_real_distribution{-world_size, world_size};

    for (u32 idx = 0; idx < light_count; ++idx) {
        auto t = static_cast<float>(idx) / static_cast<float>(light_count);
        auto &[position_radius, colour_intensity] = all_point_lights[idx];

        position_radius = {distrib(rng), distrib(rng), distrib(rng), 5.0F};
        colour_intensity = {t, 1.0f - t, 0.5f, 1.0f};
    }

    struct Cube {
        std::array<float, 4> position_radius;
        std::array<float, 4> colour_intensity;
    };

    auto cubes = std::vector<Cube>(opts.light_count / 10);
    for (u32 idx = 0; idx < cubes.size(); ++idx) {
        auto t = static_cast<float>(idx) / static_cast<float>(cubes.size());
        auto &[position_radius, colour_intensity] = cubes[idx];

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

    auto cubes_handle =
            ctx.buffers.create(Buffer::from_slice<Cube>(allocator,
                                                        VkBufferCreateInfo{
                                                                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                        },
                                                        VmaAllocationCreateInfo{}, cubes, "cubes")
                                       .value());

    VkDrawMeshTasksIndirectCommandEXT cube_meshlet_cmd{};
    cube_meshlet_cmd.groupCountX = (uint32_t(cubes.size()) + 15) / 16; // 16 threads per group
    cube_meshlet_cmd.groupCountY = 1;
    cube_meshlet_cmd.groupCountZ = 1;

    auto cube_meshlet_indirect_buffer_handle = ctx.buffers.create(
            Buffer::from_slice<VkDrawMeshTasksIndirectCommandEXT>(
                    allocator,
                    VkBufferCreateInfo{
                            .usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    },
                    VmaAllocationCreateInfo{}, std::span{&cube_meshlet_cmd, 1}, "cube_draw_indirect")
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

    VkDrawIndexedIndirectCommand indirect_cmd{
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
                            .usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    },
                    VmaAllocationCreateInfo{}, std::span{&indirect_cmd, 1}, "light_draw_indirect")
                    .value());

    VkDrawMeshTasksIndirectCommandEXT meshlet_cmd{.groupCountX = 0, .groupCountY = 0, .groupCountZ = 0};

    auto meshlet_indirect_buffer_handle = ctx.buffers.create(
            Buffer::from_slice<VkDrawMeshTasksIndirectCommandEXT>(
                    allocator,
                    VkBufferCreateInfo{
                            .usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    },
                    VmaAllocationCreateInfo{}, std::span{&meshlet_cmd, 1}, "light_draw_indirect_meshlet")
                    .value());

    glm::vec3 camera_pos = glm::vec3(0, 50, -100);
    glm::vec3 camera_target = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 camera_up = glm::vec3(0.0f, 1.0f, 0.0f);

    glm::mat4 view = glm::lookAt(camera_pos, camera_target, camera_up);
    glm::mat4 projection = glm::perspective(glm::radians(90.0f), // FOV
                                            static_cast<float>(opts.width) / static_cast<float>(opts.height),
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
    auto indirect_point_light_addr = ctx.device_address(indirect_buffer_handle);
    auto indirect_meshlet_addr = ctx.device_address(meshlet_indirect_buffer_handle);

    auto cube_meshlet_indirect_buf_addr = ctx.device_address(cube_meshlet_indirect_buffer_handle);
    auto cube_meshlet_indirect_buf = ctx.buffers.get(cube_meshlet_indirect_buffer_handle)->buffer();
    auto cube_buffer_addr = ctx.device_address(cubes_handle);
    auto indirect_buf = ctx.buffers.get(indirect_buffer_handle)->buffer();
    auto indirect_meshlet_buf = ctx.buffers.get(meshlet_indirect_buffer_handle)->buffer();

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

    struct WindowData {
        bool resized{false};
    } wd;
    auto current_extent = [](GLFWwindow *win) -> VkExtent2D {
        int fbw{0};
        int fbh{0};
        glfwGetFramebufferSize(win, &fbw, &fbh);
        return VkExtent2D{.width = static_cast<u32>(std::max(fbw, 0)), .height = static_cast<u32>(std::max(fbh, 0))};
    };

    glfwSetWindowUserPointer(window, &wd);
    glfwSetKeyCallback(window, [](auto w, auto k, auto, auto, auto) {
        if (k == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(w, GLFW_TRUE);
        }
    });
    glfwSetWindowSizeCallback(window, [](auto w, auto, auto) {
        auto &data = *static_cast<WindowData *>(glfwGetWindowUserPointer(w));
        data.resized = true;
    });
    glfwSetFramebufferSizeCallback(window, [](auto w, auto, auto) {
        auto &data = *static_cast<WindowData *>(glfwGetWindowUserPointer(w));
        data.resized = true;
    });

    glfwShowWindow(window);

    VkExtent2D last_extent = current_extent(window);
    ResizeGraph resize_graph{};
    {
        const auto swapchain_node = resize_graph.add_node("swapchain", [&](VkExtent2D, const ResizeContext &) {
            // no-op: you already call swapchain.recreate(extent) outside the graph
            // (you can move swapchain.recreate into the graph later if you want)
        });

        const auto tonemapped_node =
                resize_graph.add_node("tonemapped_image", [&](VkExtent2D e, const ResizeContext &resize_context) {
                    const auto old_tonemap = tonemapped_target_handle;


                    tonemapped_target_handle = ctx.create_texture(create_offscreen_target(
                            allocator, e.width, e.height, VK_FORMAT_R8G8B8A8_UNORM, "tonemapped"));
                    destroy(ctx, old_tonemap, resize_context.retire_value);
                });

        const auto offscreen_node =
                resize_graph.add_node("offscreen_targets", [&](VkExtent2D e, const ResizeContext &resize_ctx) {
                    if (e.width == 0 || e.height == 0)
                        return;

                    const auto old_color = offscreen_target_handle;
                    const auto old_depth = offscreen_depth_target_handle;

                    offscreen_target_handle = ctx.create_texture(create_offscreen_target(
                            allocator, e.width, e.height, VK_FORMAT_R32G32B32A32_SFLOAT, "offscreen"));

                    offscreen_depth_target_handle = ctx.create_texture(
                            create_depth_target(allocator, e.width, e.height, VK_FORMAT_D32_SFLOAT, "offscreen_depth"));


                    destroy(ctx, old_color, resize_ctx.retire_value);
                    destroy(ctx, old_depth, resize_ctx.retire_value);
                });

        const auto uniforms_node = resize_graph.add_node("frame_ubo_camera", [&](VkExtent2D e, const ResizeContext &) {
            if (e.width == 0 || e.height == 0)
                return;

            const float aspect_ratio = static_cast<float>(e.width) / static_cast<float>(e.height);

            ubo_data.projection = glm::perspective(glm::radians(90.0f), aspect_ratio, 0.1f, 1000.0f);
            ubo_data.inv_projection = glm::inverse(ubo_data.projection);
            ubo_data.view_projection = ubo_data.projection * ubo_data.view;
            const auto planes = extract_frustum_planes(ubo_data.inv_projection);
            ubo_data.frustum_planes = {planes[0], planes[1], planes[2], planes[3], planes[4], planes[5]};

            // You likely upload this later via staging; if not, do it here.
            // (I don’t see your UBO upload path in the snippet, so leaving it as “update data”.)
        });


        resize_graph.add_dependency(tonemapped_node, offscreen_node);
        resize_graph.add_dependency(offscreen_node, swapchain_node);
        resize_graph.add_dependency(uniforms_node, swapchain_node);
    }
    if (auto str = resize_graph.to_graphviz_dot(); !str.empty()) {
        std::ofstream output{"graph.dot", std::ios::out};
        output.write(str.c_str(), str.size());
    }

    u64 frame_index{};
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        throttle(tl_compute, device);
        throttle(tl_graphics, device);

        const u64 completed_now = std::min(tl_compute.completed, tl_graphics.completed);

        if (const auto extent = current_extent(window);
            extent.width != last_extent.width || extent.height != last_extent.height) {
            if (extent.width == 0 || extent.height == 0) {
                // Minimized window, skip until un-minimized
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            last_extent = extent;

            if (auto r = swapchain.recreate(extent); !r) {
                vk_check(r.error());
            }

            resize_graph.rebuild(extent, ResizeContext{
                                                 .allocator = allocator,
                                                 .retire_value = completed_now,
                                         });
        }

        const auto frame_extent = swapchain.extent();
        auto start_time = std::chrono::high_resolution_clock::now();

#ifdef HAS_RENDERDOC
        if (rdoc_api)
            rdoc_api->StartFrameCapture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(instance.instance), NULL);
#endif
        bindless.repopulate_if_needed(ctx.textures, ctx.samplers);

        const auto bounded_frame_index = static_cast<u32>(frame_index % frames_in_flight);
        auto &fs = frames[bounded_frame_index];

        if (fs.frame_done_value > 0) {
            VkSemaphoreWaitInfo wi{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                   .pNext = nullptr,
                                   .flags = 0,
                                   .semaphoreCount = 1,
                                   .pSemaphores = &tl_graphics.timeline,
                                   .pValues = &fs.frame_done_value};
            vk_check(vkWaitSemaphores(device, &wi, UINT64_MAX));

            if (auto ms = read_timestamp_ms(compute_query_pool[bounded_frame_index])) {
                gpu_compute_ms.add_sample(*ms);
            }
            if (auto ms = read_timestamp_ms(graphics_query_pool[bounded_frame_index])) {
                gpu_graphics_ms.add_sample(*ms);
            }
        }

        auto acquired = swapchain.acquire_next_image(bounded_frame_index);
        if (!acquired) {
            const VkResult res = acquired.error();
            if (res == VK_ERROR_OUT_OF_DATE_KHR) {
                continue;
            }
            vk_check(res);
        }

        const auto swap_image_index = acquired->image_index;
        const auto frame_sync = acquired->sync;

        auto predepth_val = submit_stage(
                tl_graphics, device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(tracy_graphics.ctx, cmd, "Predepth");
                    auto &&depth = ctx.textures.get(offscreen_depth_target_handle);

                    depth->transition_if_not_initialised(
                            cmd, VK_IMAGE_LAYOUT_GENERAL,
                            {VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT});

                    VkRenderingAttachmentInfo depth_attachment{
                            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                            .imageView = depth->sampled_view,
                            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                            .clearValue = {.depthStencil = {0.0f, 0}},
                    };

                    VkRenderingInfo rendering_info{
                            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                            .renderArea = {.offset = {0, 0}, .extent = {frame_extent.width, frame_extent.height}},
                            .layerCount = 1,
                            .pDepthAttachment = &depth_attachment,
                    };

                    vkCmdBeginRendering(cmd, &rendering_info);
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, predepth_pipeline.pipeline);

                    PredepthPushConstants pc = {
                            .ubo = frame_ubo,
                            .cube_buffer = cube_buffer_addr,
                            .indirect_meshlet = cube_meshlet_indirect_buf_addr,
                    };

                    VkViewport vp{
                            .x = 0,
                            .y = static_cast<float>(frame_extent.height),
                            .width = static_cast<float>(frame_extent.width),
                            .height = -static_cast<float>(frame_extent.height),
                            .minDepth = 1.0f,
                            .maxDepth = 0.0f,
                    };

                    VkRect2D sc{.offset = {0, 0}, .extent = {frame_extent.width, frame_extent.height}};
                    vkCmdSetViewport(cmd, 0, 1, &vp);
                    vkCmdSetScissor(cmd, 0, 1, &sc);
                    vkCmdPushConstants(cmd, predepth_pipeline.layout,
                                       VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TASK_BIT_EXT |
                                               VK_SHADER_STAGE_MESH_BIT_EXT,
                                       0, sizeof(pc), &pc);

                    draw_mesh(cmd, cube_meshlet_indirect_buf, VkDeviceSize{0}, 1u,
                              static_cast<u32>(sizeof(VkDrawMeshTasksIndirectCommandEXT)));

                    vkCmdEndRendering(cmd);
                },
                no_waits);
        fs.timeline_values[stage_index(Stage::Predepth)] = predepth_val;

        const std::array culling_waits{TimelineWait{.value = fs.timeline_values[stage_index(Stage::Predepth)],
                                                    .semaphore = tl_graphics.timeline,
                                                    .stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT}};
        auto light_val = submit_stage(
                tl_compute, device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(tracy_compute.ctx, cmd, "LightCulling");
                    const auto *cqs = ctx.query_pools.get(compute_query_pool[bounded_frame_index]);
                    const auto &cqp = cqs->pool;

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
                            .indirect_point_light = indirect_point_light_addr,
                            .indirect_meshlet = indirect_meshlet_addr,
                            .image_index = offscreen_target_handle.index(),
                            .light_count = light_count,
                            .group_count = group_count,
                    };


                    auto bind_and_dispatch = [&](auto &pl, u32 groups_x) {
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.layout, 0, 1, &bindless.set, 0,
                                                nullptr);

                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.pipeline);

                        vkCmdPushConstants(cmd, pl.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GpuPushConstants),
                                           &pc);

                        vkCmdDispatch(cmd, groups_x, 1u, 1u);
                    };

                    VkMemoryBarrier2 mem_barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                                 .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                 .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                                                 .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                 .dstAccessMask =
                                                         VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT};
                    VkDependencyInfo dep_info{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                              .memoryBarrierCount = 1,
                                              .pMemoryBarriers = &mem_barrier};

                    fill_zeros(cmd, ctx.buffers, point_light_handle, flags_handle, prefix_handle, group_sums_handle,
                               group_offsets_handle, compact_lights_handle);
                    vkCmdPipelineBarrier2(cmd, &dep_info);

                    bind_and_dispatch(flags, group_count);
                    vkCmdPipelineBarrier2(cmd, &dep_info);

                    bind_and_dispatch(scan_local, group_count);
                    vkCmdPipelineBarrier2(cmd, &dep_info);

                    bind_and_dispatch(scan_groups, group_count);
                    vkCmdPipelineBarrier2(cmd, &dep_info);

                    bind_and_dispatch(compact, group_count);
                    vkCmdPipelineBarrier2(cmd, &dep_info);

                    bind_and_dispatch(setup_indirect, 1u);

                    std::array<VkBufferMemoryBarrier2, 2> indirect_barrier{
                            VkBufferMemoryBarrier2{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                                                   .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                   .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                                                   .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                                   .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                   .buffer = indirect_buf,
                                                   .offset = 0,
                                                   .size = VK_WHOLE_SIZE},
                            VkBufferMemoryBarrier2{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                                                   .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                   .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                                                   .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                                   .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                   .buffer = indirect_meshlet_buf,
                                                   .offset = 0,
                                                   .size = VK_WHOLE_SIZE}};

                    VkDependencyInfo indirect_dep{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                                  .bufferMemoryBarrierCount = 2,
                                                  .pBufferMemoryBarriers = indirect_barrier.data()};

                    vkCmdPipelineBarrier2(cmd, &indirect_dep);

                    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, cqp,
                                        static_cast<u32>(GpuStamp::End));
                    TRACY_GPU_COLLECT(tracy_compute.ctx, cmd);
                },
                SubmitSynchronisation{.timeline_waits = culling_waits});

        fs.timeline_values[stage_index(Stage::LightCulling)] = light_val;

        const std::array gbuffer_waits{
                TimelineWait{
                        .value = fs.timeline_values[stage_index(Stage::LightCulling)],
                        .semaphore = tl_compute.timeline,
                        .stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                },
        };

        auto gbuffer_val = submit_stage(
                tl_graphics, device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(tracy_graphics.ctx, cmd, "GBuffer");
                    const auto *gqs = ctx.query_pools.get(graphics_query_pool[bounded_frame_index]);
                    auto &&[offscreen, depth] =
                            ctx.textures.get_multiple(offscreen_target_handle, offscreen_depth_target_handle);
                    const VkQueryPool &gqp = gqs->pool;

                    vkCmdResetQueryPool(cmd, gqp, 0, gqs->query_count);
                    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, gqp, 0);

                    offscreen->transition_if_not_initialised(
                            cmd, VK_IMAGE_LAYOUT_GENERAL,
                            {VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT});

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
                            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                            .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                    };

                    VkRenderingInfo rendering_info{
                            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                            .renderArea = {.offset = {0, 0}, .extent = {frame_extent.width, frame_extent.height}},
                            .layerCount = 1,
                            .colorAttachmentCount = 1,
                            .pColorAttachments = &color_attachment,
                            .pDepthAttachment = &depth_attachment,
                    };

                    vkCmdBeginRendering(cmd, &rendering_info);

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, point_light_pipeline.pipeline);

                    const RenderingPushConstants pc{
                            .ubo = frame_ubo,
                            .cubes = cube_buffer_addr,
                            .indirect_meshlet = cube_meshlet_indirect_buf_addr,
                    };

                    VkViewport vp{
                            .x = 0,
                            .y = static_cast<float>(frame_extent.height),
                            .width = static_cast<float>(frame_extent.width),
                            .height = -static_cast<float>(frame_extent.height),
                            .minDepth = 1.0f,
                            .maxDepth = 0.0f,
                    };

                    VkRect2D sc{.offset = {0, 0}, .extent = {frame_extent.width, frame_extent.height}};

                    vkCmdSetViewport(cmd, 0, 1, &vp);
                    vkCmdSetScissor(cmd, 0, 1, &sc);
                    vkCmdPushConstants(cmd, point_light_pipeline.layout,
                                       VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_MESH_BIT_EXT |
                                               VK_SHADER_STAGE_TASK_BIT_EXT,
                                       0, sizeof(pc), &pc);

                    draw_mesh(cmd, cube_meshlet_indirect_buf, VkDeviceSize{0}, 1u,
                              static_cast<u32>(sizeof(VkDrawMeshTasksIndirectCommandEXT)));

                    vkCmdEndRendering(cmd);

                    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, gqp, 1);
                },
                SubmitSynchronisation{.timeline_waits = gbuffer_waits});
        fs.timeline_values[stage_index(Stage::GBuffer)] = gbuffer_val;

        const std::array tonemap_waits{
                TimelineWait{
                        .value = fs.timeline_values[stage_index(Stage::GBuffer)],
                        .semaphore = tl_graphics.timeline,
                        .stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                },
        };

        auto tonemap_val = submit_stage(
                tl_graphics, device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(tracy_graphics.ctx, cmd, "Tonemapping");
                    auto &&hdr = ctx.textures.get(offscreen_target_handle);
                    auto &&ldr = ctx.textures.get(tonemapped_target_handle);

                    // Transition HDR for sampling
                    hdr->transition_if_not_initialised(
                            cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            {VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT});

                    // Transition LDR for rendering
                    ldr->transition_if_not_initialised(
                            cmd, VK_IMAGE_LAYOUT_GENERAL,
                            {VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT});

                    VkRenderingAttachmentInfo color_attachment{
                            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                            .imageView = ldr->sampled_view,
                            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                            .clearValue = {.color = {0, 0, 0, 1}},
                    };

                    VkRenderingInfo ri{
                            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                            .renderArea = {.offset = {0, 0}, .extent = {frame_extent.width, frame_extent.height}},
                            .layerCount = 1,
                            .colorAttachmentCount = 1,
                            .pColorAttachments = &color_attachment,
                    };

                    vkCmdBeginRendering(cmd, &ri);

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tonemap_pipeline.pipeline);

                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tonemap_pipeline.layout, 0, 1,
                                            &bindless.set, 0, nullptr);

                    float exposure = 1.0f;
                    TonemapPushConstants pc{
                            .exposure = exposure,
                            .image_index = offscreen_target_handle.index(),
                            .sampler_index = 0,
                    };


                    VkViewport vp{
                            .x = 0,
                            .y = static_cast<float>(frame_extent.height),
                            .width = static_cast<float>(frame_extent.width),
                            .height = -static_cast<float>(frame_extent.height),
                            .minDepth = 1.0f,
                            .maxDepth = 0.0f,
                    };

                    VkRect2D sc{.offset = {0, 0}, .extent = {frame_extent.width, frame_extent.height}};
                    vkCmdSetViewport(cmd, 0, 1, &vp);
                    vkCmdSetScissor(cmd, 0, 1, &sc);

                    vkCmdPushConstants(cmd, tonemap_pipeline.layout,
                                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

                    vkCmdDraw(cmd, 3, 1, 0, 0); // fullscreen triangle

                    vkCmdEndRendering(cmd);
                },
                SubmitSynchronisation{.timeline_waits = tonemap_waits});

        fs.timeline_values[stage_index(Stage::Tonemapping)] = tonemap_val;

        const std::array present_timeline_waits{
                TimelineWait{
                        .value = fs.timeline_values[stage_index(Stage::Tonemapping)],
                        .semaphore = tl_graphics.timeline,
                },
        };

        const std::array present_binary_waits{BinaryWait{
                .semaphore = frame_sync.image_available,
                .stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        }};

        const std::array present_binary_signals{frame_sync.render_finished};

        auto swapchain_val = submit_stage(
                tl_graphics, device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(tracy_graphics.ctx, cmd, "CopyToSwapchain");
                    auto &&tonemapped = ctx.textures.get(tonemapped_target_handle);

                    const auto dst_image = swapchain.image(swap_image_index);
                    const auto src_image = tonemapped->image;

                    const std::array barriers{VkImageMemoryBarrier2{
                                                      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                                      .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                                      .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                                      .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                      .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                                                      .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                                      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                      .image = src_image,
                                                      .subresourceRange =
                                                              VkImageSubresourceRange{
                                                                      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                                      .baseMipLevel = 0,
                                                                      .levelCount = 1,
                                                                      .baseArrayLayer = 0,
                                                                      .layerCount = 1,
                                                              },
                                              },
                                              VkImageMemoryBarrier2{
                                                      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                                      .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                                                      .srcAccessMask = VK_ACCESS_2_NONE,
                                                      .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                      .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                                      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                      .image = dst_image,
                                                      .subresourceRange =
                                                              VkImageSubresourceRange{
                                                                      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                                      .baseMipLevel = 0,
                                                                      .levelCount = 1,
                                                                      .baseArrayLayer = 0,
                                                                      .layerCount = 1,
                                                              },
                                              }};

                    VkDependencyInfo dep_info{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                              .imageMemoryBarrierCount = static_cast<u32>(barriers.size()),
                                              .pImageMemoryBarriers = barriers.data()};

                    vkCmdPipelineBarrier2(cmd, &dep_info);

                    VkImageCopy region{
                            .srcSubresource =
                                    VkImageSubresourceLayers{
                                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                            .mipLevel = 0,
                                            .baseArrayLayer = 0,
                                            .layerCount = 1,
                                    },
                            .srcOffset = VkOffset3D{0, 0, 0},
                            .dstSubresource =
                                    VkImageSubresourceLayers{
                                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                            .mipLevel = 0,
                                            .baseArrayLayer = 0,
                                            .layerCount = 1,
                                    },
                            .dstOffset = VkOffset3D{0, 0, 0},
                            .extent = VkExtent3D{frame_extent.width, frame_extent.height, 1},
                    };

                    vkCmdCopyImage(cmd, src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst_image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

                    const std::array end_barriers{
                            VkImageMemoryBarrier2{
                                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                    .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                    .srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                                    .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                    .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                    .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .image = src_image,
                                    .subresourceRange =
                                            VkImageSubresourceRange{
                                                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                    .baseMipLevel = 0,
                                                    .levelCount = 1,
                                                    .baseArrayLayer = 0,
                                                    .layerCount = 1,
                                            },
                            },
                            VkImageMemoryBarrier2{
                                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                    .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                    .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                    .dstStageMask = VK_PIPELINE_STAGE_2_NONE,
                                    .dstAccessMask = VK_ACCESS_2_NONE,
                                    .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .image = dst_image,
                                    .subresourceRange =
                                            VkImageSubresourceRange{
                                                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                    .baseMipLevel = 0,
                                                    .levelCount = 1,
                                                    .baseArrayLayer = 0,
                                                    .layerCount = 1,
                                            },
                            },
                    };

                    VkDependencyInfo end_dep_info{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                                  .imageMemoryBarrierCount = static_cast<u32>(end_barriers.size()),
                                                  .pImageMemoryBarriers = end_barriers.data()};

                    vkCmdPipelineBarrier2(cmd, &end_dep_info);
                    TRACY_GPU_COLLECT(tracy_graphics.ctx, cmd);
                },
                SubmitSynchronisation{
                        .timeline_waits = present_timeline_waits,
                        .binary_waits = present_binary_waits,
                        .binary_signals = present_binary_signals,
                });

        fs.frame_done_value = swapchain_val;

        const auto completed = std::min(tl_compute.completed, tl_graphics.completed);
        ctx.destroy_queue.retire(completed);
#ifdef HAS_RENDERDOC
        if (rdoc_api)
            rdoc_api->EndFrameCapture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(instance.instance), NULL);
#endif
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(frame_end - start_time).count();
        stats.add_sample(ms);

        const VkResult present_res = swapchain.present(graphics_queue, swap_image_index, frame_sync.render_finished);
        if (present_res == VK_ERROR_OUT_OF_DATE_KHR || present_res == VK_SUBOPTIMAL_KHR) {
            auto result = swapchain.recreate(current_extent(window));
            if (!result)
                vk_check(result.error());
        } else {
            vk_check(present_res);
        }
        FrameMark;


        frame_index++;
    }

    info("Light count {}", opts.light_count);
    info("frames: {}", stats.samples.size());
    info("mean/frametime:   {:.3f} ms", stats.mean);
    info("median: {:.3f} ms", stats.median());
    info("stddev: {:.3f} ms", stats.stddev_sample());
    info("quartiles: {}", stats.quartiles());
    info("Total: {:.3f} s", stats.total() / 1000.0F);

    info("GPU compute mean:   {:.3f} ms", gpu_compute_ms.avg());
    info("GPU compute p95:    {:.3f} ms", gpu_compute_ms.p95());
    info("GPU graphics mean:  {:.3f} ms", gpu_graphics_ms.avg());
    info("GPU graphics p95:   {:.3f} ms", gpu_graphics_ms.p95());

    const auto &&[oth, ph] = ctx.textures.get_multiple(offscreen_target_handle, perlin_handle);
    {
        ZoneScopedNC("batch_write_images", 0xFF00AA);
        std::array requests{image_operations::ImageWriteRequest{oth, "output.bmp"},
                            image_operations::ImageWriteRequest{ph, "perlin.bmp"}};
        image_operations::write_batch_to_disk(allocator, requests);
    }

    pipeline_cache.reset();
    ctx.clear_all();

    ctx.destroy_queue.retire(UINT64_MAX);

    tracy_compute.shutdown();
    tracy_graphics.shutdown();

    destruction::pipeline(device, flags, scan_local, scan_groups, compact, setup_indirect, point_light_pipeline,
                          predepth_pipeline, tonemap_pipeline);
    destruction::global_command_context(command_context);
    destruction::bindless_set(device, bindless);
    destruction::timeline_compute(device, tl_graphics);
    destruction::timeline_compute(device, tl_compute);
    destruction::allocator(allocator);
    swapchain.destroy();
    destruction::wsi(instance.instance, surface, window);
    vkDeviceWaitIdle(device);
    destruction::device(device);
    destruction::instance(instance);

    return 0;
}


auto main(int argc, char **argv) -> int {
    execute(argc, argv);

    info("Bindless headless setup and teardown completed successfully.");
    return 0;
}
