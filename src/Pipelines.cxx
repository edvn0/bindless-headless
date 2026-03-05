#include "Pipelines.hxx"
#include "BindlessHeadless.hxx"
#include "Mesh.hxx"
#include "PipelineCache.hxx"
#include "RenderContext.hxx"

#include <glm/glm.hpp>
#include <ranges>
#include <utility>


auto create_compute_pipeline(VkDevice device, PipelineCache *cache, VkDescriptorSetLayout layout,
                             const std::vector<u32> &code, std::size_t push_constant_size,
                             const std::string_view entry_name) -> CompiledPipeline {
    VkShaderModule compute_shader{};
    auto ci = create_info<VkShaderModuleCreateInfo>();
    ci.codeSize = code.size() * sizeof(u32);
    ci.pCode = code.data();
    vk_check(vkCreateShaderModule(device, &ci, nullptr, &compute_shader));

    VkPushConstantRange push_constant_range{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = static_cast<u32>(push_constant_size),
    };

    VkPipelineLayout pi_layout{};
    auto plci = create_info<VkPipelineLayoutCreateInfo>();
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &push_constant_range;
    vk_check(vkCreatePipelineLayout(device, &plci, nullptr, &pi_layout));
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, pi_layout, entry_name);

    // u32, u32
    const std::array<u32, 2> data{MAX_WAVES_PER_GROUP, THREADS_PER_GROUP};

    const VkSpecializationMapEntry waves_per_group_spec_map_entry{
            .constantID = 0,
            .offset = 0,
            .size = sizeof(u32),
    };
    const VkSpecializationMapEntry threads_per_group_spec_map_entry{
            .constantID = 1,
            .offset = sizeof(u32),
            .size = sizeof(u32),
    };
    const std::array entries{waves_per_group_spec_map_entry, threads_per_group_spec_map_entry};
    VkSpecializationInfo spec_info{};
    spec_info.mapEntryCount = 2;
    spec_info.pMapEntries = entries.data();
    spec_info.dataSize = 2 * sizeof(u32);
    spec_info.pData = data.data();

    const std::array spec_infos{spec_info};

    auto stage_ci = create_info<VkPipelineShaderStageCreateInfo>();
    stage_ci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_ci.module = compute_shader;
    stage_ci.pName = entry_name.data();
    stage_ci.pSpecializationInfo = spec_infos.data();

    auto cpci = create_info<VkComputePipelineCreateInfo>();
    cpci.stage = stage_ci;
    cpci.layout = pi_layout;
    cpci.basePipelineHandle = VK_NULL_HANDLE;
    cpci.basePipelineIndex = -1;

    VkPipeline pipeline{VK_NULL_HANDLE};
    VkPipelineCache cache_handle = cache ? cache->get() : VK_NULL_HANDLE;
    vk_check(vkCreateComputePipelines(device, cache_handle, 1, &cpci, nullptr, &pipeline));
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, entry_name);

    vkDestroyShaderModule(device, compute_shader, nullptr);
    return {pipeline, pi_layout};
}

auto create_predepth_pipeline(VkDevice device, PipelineCache *cache, VkDescriptorSetLayout bindless_layout,
                              const std::vector<u32> &vert_code, VkFormat depth_format, VkSampleCountFlagBits samples)
        -> CompiledPipeline {
    VkShaderModule vert_module{};
    auto shader_ci = create_info<VkShaderModuleCreateInfo>();
    shader_ci.codeSize = vert_code.size() * sizeof(u32);
    shader_ci.pCode = vert_code.data();
    vk_check(vkCreateShaderModule(device, &shader_ci, nullptr, &vert_module));

    auto vert_stage_ci = create_info<VkPipelineShaderStageCreateInfo>();
    vert_stage_ci.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_stage_ci.module = vert_module;
    vert_stage_ci.pName = "main_vs_mdi";


    std::array stages = {
            vert_stage_ci,
    };

    // 2. Pipeline Layout (Inherit bindless + push constants)
    VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .offset = 0,
            .size = sizeof(PredepthPushConstants),
    };

    auto layout_ci = create_info<VkPipelineLayoutCreateInfo>();
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &bindless_layout;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &push_range;

    VkPipelineLayout layout;
    vkCreatePipelineLayout(device, &layout_ci, nullptr, &layout);
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, layout, "predepth");

    // 3. Specialized Depth State
    auto ds = create_info<VkPipelineDepthStencilStateCreateInfo>();
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL; // Reverse-Z: Near is 1.0, Far is 0.0
    ds.minDepthBounds = 0.0f;
    ds.maxDepthBounds = 1.0f;

    // 4. No Color Attachments (The secret to Pre-Depth speed)
    auto cb = create_info<VkPipelineColorBlendStateCreateInfo>();
    cb.attachmentCount = 0;
    cb.pAttachments = nullptr;

    // 5. Rasterization (Ensure Back-Face Culling is ON)
    auto rs = create_info<VkPipelineRasterizationStateCreateInfo>();
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    // 6. Dynamic Rendering Info
    auto rendering_info = create_info<VkPipelineRenderingCreateInfo>();
    rendering_info.depthAttachmentFormat = depth_format;

    // Viewport/Scissor setup (Standard)
    auto vp = create_info<VkPipelineViewportStateCreateInfo>();
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    auto ms = create_info<VkPipelineMultisampleStateCreateInfo>();
    ms.rasterizationSamples = samples;

    std::array dynamic_states = {VK_DYNAMIC_STATE_VIEWPORT,         VK_DYNAMIC_STATE_SCISSOR,
                                 VK_DYNAMIC_STATE_DEPTH_COMPARE_OP, VK_DYNAMIC_STATE_DEPTH_BOUNDS,
                                 VK_DYNAMIC_STATE_CULL_MODE,        VK_DYNAMIC_STATE_FRONT_FACE};
    auto dy = create_info<VkPipelineDynamicStateCreateInfo>();
    dy.dynamicStateCount = static_cast<u32>(dynamic_states.size());
    dy.pDynamicStates = dynamic_states.data();

    std::array<VkVertexInputBindingDescription, 1> binding_descriptions{VkVertexInputBindingDescription{
            .binding = 0,
            .stride = sizeof(PositionOnlyVertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    }};

    std::array<VkVertexInputAttributeDescription, 2> attribute_descriptions{
            VkVertexInputAttributeDescription{
                    .location = 0,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(PositionOnlyVertex, pos),
            },
            VkVertexInputAttributeDescription{
                    .location = 1,
                    .binding = 0,
                    .format = VK_FORMAT_R32_UINT,
                    .offset = offsetof(PositionOnlyVertex, uvs),
            }};

    auto vertex_input = create_info<VkPipelineVertexInputStateCreateInfo>();
    vertex_input.vertexBindingDescriptionCount = static_cast<u32>(binding_descriptions.size());
    vertex_input.pVertexBindingDescriptions = binding_descriptions.data();
    vertex_input.vertexAttributeDescriptionCount = static_cast<u32>(attribute_descriptions.size());
    vertex_input.pVertexAttributeDescriptions = attribute_descriptions.data();

    auto assembly_state = create_info<VkPipelineInputAssemblyStateCreateInfo>();
    assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    assembly_state.primitiveRestartEnable = VK_FALSE;

    auto ci = create_info<VkGraphicsPipelineCreateInfo>();
    ci.pNext = &rendering_info;
    ci.stageCount = static_cast<u32>(stages.size());
    ci.pStages = stages.data();
    ci.pVertexInputState = &vertex_input;
    ci.pInputAssemblyState = &assembly_state;
    ci.pViewportState = &vp;
    ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms;
    ci.pDepthStencilState = &ds;
    ci.pColorBlendState = &cb;
    ci.pDynamicState = &dy;
    ci.layout = layout;
    ci.basePipelineHandle = VK_NULL_HANDLE;
    ci.basePipelineIndex = -1;

    VkPipeline pipeline;
    VkPipelineCache cache_handle = cache ? cache->get() : VK_NULL_HANDLE;
    vkCreateGraphicsPipelines(device, cache_handle, 1, &ci, nullptr, &pipeline);
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, "predepth");

    // Cleanup local modules
    vkDestroyShaderModule(device, vert_module, nullptr);

    return {.pipeline = pipeline, .layout = layout};
}

auto create_predepth_pipeline(VkDevice device, PipelineCache *cache, VkDescriptorSetLayout bindless_layout,
                              const std::vector<u32> &vert_code, const std::vector<u32> &frag_code,
                              VkFormat depth_format, VkSampleCountFlagBits samples) -> CompiledPipeline {
    VkShaderModule vert_module{};
    auto shader_ci = create_info<VkShaderModuleCreateInfo>();
    shader_ci.codeSize = vert_code.size() * sizeof(u32);
    shader_ci.pCode = vert_code.data();
    vk_check(vkCreateShaderModule(device, &shader_ci, nullptr, &vert_module));

    VkShaderModule frag_module{};
    shader_ci.codeSize = frag_code.size() * sizeof(u32);
    shader_ci.pCode = frag_code.data();
    vk_check(vkCreateShaderModule(device, &shader_ci, nullptr, &frag_module));

    auto vert_stage_ci = create_info<VkPipelineShaderStageCreateInfo>();
    vert_stage_ci.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_stage_ci.module = vert_module;
    vert_stage_ci.pName = "main_vs_mdi";

    auto frag_stage_ci = create_info<VkPipelineShaderStageCreateInfo>();
    frag_stage_ci.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_stage_ci.module = frag_module;
    frag_stage_ci.pName = "fs_main";
    std::array stages = {
            vert_stage_ci,
            frag_stage_ci,
    };

    // 2. Pipeline Layout (Inherit bindless + push constants)
    VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(PredepthPushConstants),
    };

    auto layout_ci = create_info<VkPipelineLayoutCreateInfo>();
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &bindless_layout;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &push_range;

    VkPipelineLayout layout;
    vkCreatePipelineLayout(device, &layout_ci, nullptr, &layout);
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, layout, "predepth_alpha_tested");

    // 3. Specialized Depth State
    auto ds = create_info<VkPipelineDepthStencilStateCreateInfo>();
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL; // Reverse-Z: Near is 1.0, Far is 0.0
    ds.minDepthBounds = 0.0f;
    ds.maxDepthBounds = 1.0f;

    // 4. No Color Attachments (The secret to Pre-Depth speed)
    auto cb = create_info<VkPipelineColorBlendStateCreateInfo>();
    cb.attachmentCount = 0;
    cb.pAttachments = nullptr;

    // 5. Rasterization (Ensure Back-Face Culling is ON)
    auto rs = create_info<VkPipelineRasterizationStateCreateInfo>();
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    // 6. Dynamic Rendering Info
    auto rendering_info = create_info<VkPipelineRenderingCreateInfo>();
    rendering_info.depthAttachmentFormat = depth_format;

    // Viewport/Scissor setup (Standard)
    auto vp = create_info<VkPipelineViewportStateCreateInfo>();
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    auto ms = create_info<VkPipelineMultisampleStateCreateInfo>();
    ms.rasterizationSamples = samples;

    std::array dynamic_states = {VK_DYNAMIC_STATE_VIEWPORT,         VK_DYNAMIC_STATE_SCISSOR,
                                 VK_DYNAMIC_STATE_DEPTH_COMPARE_OP, VK_DYNAMIC_STATE_DEPTH_BOUNDS,
                                 VK_DYNAMIC_STATE_CULL_MODE,        VK_DYNAMIC_STATE_FRONT_FACE};
    auto dy = create_info<VkPipelineDynamicStateCreateInfo>();
    dy.dynamicStateCount = static_cast<u32>(dynamic_states.size());
    dy.pDynamicStates = dynamic_states.data();

    std::array<VkVertexInputBindingDescription, 1> binding_descriptions{VkVertexInputBindingDescription{
            .binding = 0,
            .stride = sizeof(PositionOnlyVertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    }};

    std::array<VkVertexInputAttributeDescription, 2> attribute_descriptions{
            VkVertexInputAttributeDescription{
                    .location = 0,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(PositionOnlyVertex, pos),

            },
            VkVertexInputAttributeDescription{
                    .location = 1,
                    .binding = 0,
                    .format = VK_FORMAT_R32_UINT,
                    .offset = offsetof(PositionOnlyVertex, uvs),
            },
    };

    auto vertex_input = create_info<VkPipelineVertexInputStateCreateInfo>();
    vertex_input.vertexBindingDescriptionCount = static_cast<u32>(binding_descriptions.size());
    vertex_input.pVertexBindingDescriptions = binding_descriptions.data();
    vertex_input.vertexAttributeDescriptionCount = static_cast<u32>(attribute_descriptions.size());
    vertex_input.pVertexAttributeDescriptions = attribute_descriptions.data();

    auto assembly_state = create_info<VkPipelineInputAssemblyStateCreateInfo>();
    assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    assembly_state.primitiveRestartEnable = VK_FALSE;

    auto ci = create_info<VkGraphicsPipelineCreateInfo>();
    ci.pNext = &rendering_info;
    ci.stageCount = static_cast<u32>(stages.size());
    ci.pStages = stages.data();
    ci.pVertexInputState = &vertex_input;
    ci.pInputAssemblyState = &assembly_state;
    ci.pViewportState = &vp;
    ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms;
    ci.pDepthStencilState = &ds;
    ci.pColorBlendState = &cb;
    ci.pDynamicState = &dy;
    ci.layout = layout;
    ci.basePipelineHandle = VK_NULL_HANDLE;
    ci.basePipelineIndex = -1;

    VkPipeline pipeline;
    VkPipelineCache cache_handle = cache ? cache->get() : VK_NULL_HANDLE;

    vkCreateGraphicsPipelines(device, cache_handle, 1, &ci, nullptr, &pipeline);
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, "predepth_alpha_tested");

    // Cleanup local modules
    vkDestroyShaderModule(device, vert_module, nullptr);
    vkDestroyShaderModule(device, frag_module, nullptr);


    return {.pipeline = pipeline, .layout = layout};
}

auto create_directional_shadow_map_pipeline(VkDevice device, PipelineCache *cache,
                                            VkDescriptorSetLayout bindless_layout, const std::vector<u32> &vert_code,
                                            const std::vector<u32> &frag_code, VkFormat depth_format,
                                            VkSampleCountFlagBits samples) -> CompiledPipeline {
    VkShaderModule vert_module{};
    auto shader_ci = create_info<VkShaderModuleCreateInfo>();
    shader_ci.codeSize = vert_code.size() * sizeof(u32);
    shader_ci.pCode = vert_code.data();
    vk_check(vkCreateShaderModule(device, &shader_ci, nullptr, &vert_module));

    VkShaderModule frag_module{};
    shader_ci.codeSize = frag_code.size() * sizeof(u32);
    shader_ci.pCode = frag_code.data();
    vk_check(vkCreateShaderModule(device, &shader_ci, nullptr, &frag_module));

    auto vert_stage_ci = create_info<VkPipelineShaderStageCreateInfo>();
    vert_stage_ci.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_stage_ci.module = vert_module;
    vert_stage_ci.pName = "shadow_vs_mdi";

    auto frag_stage_ci = create_info<VkPipelineShaderStageCreateInfo>();
    frag_stage_ci.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_stage_ci.module = frag_module;
    frag_stage_ci.pName = "shadow_fs_main";

    std::array stages = {vert_stage_ci, frag_stage_ci};

    // Pipeline Layout
    VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(ShadowMapPushConstants),
    };

    auto layout_ci = create_info<VkPipelineLayoutCreateInfo>();
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &bindless_layout;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &push_range;

    VkPipelineLayout layout;
    vkCreatePipelineLayout(device, &layout_ci, nullptr, &layout);
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, layout, "shadow_map");

    // Depth State
    auto ds = create_info<VkPipelineDepthStencilStateCreateInfo>();
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL; // Reverse-Z
    ds.minDepthBounds = 0.0f;
    ds.maxDepthBounds = 1.0f;

    // No Color Attachments (depth-only pass)
    auto cb = create_info<VkPipelineColorBlendStateCreateInfo>();
    cb.attachmentCount = 0;
    cb.pAttachments = nullptr;

    // Rasterization with depth bias for shadow acne
    auto rs = create_info<VkPipelineRasterizationStateCreateInfo>();
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    rs.depthBiasEnable = VK_TRUE;
    rs.depthBiasConstantFactor = 0.0f; // Set dynamically
    rs.depthBiasClamp = 0.0f;
    rs.depthBiasSlopeFactor = 0.0f; // Set dynamically

    // Dynamic Rendering Info
    auto rendering_info = create_info<VkPipelineRenderingCreateInfo>();
    rendering_info.depthAttachmentFormat = depth_format;

    auto vp = create_info<VkPipelineViewportStateCreateInfo>();
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    auto ms = create_info<VkPipelineMultisampleStateCreateInfo>();
    ms.rasterizationSamples = samples;

    std::array dynamic_states = {VK_DYNAMIC_STATE_VIEWPORT,         VK_DYNAMIC_STATE_SCISSOR,
                                 VK_DYNAMIC_STATE_DEPTH_COMPARE_OP, VK_DYNAMIC_STATE_DEPTH_BOUNDS,
                                 VK_DYNAMIC_STATE_CULL_MODE,        VK_DYNAMIC_STATE_FRONT_FACE,
                                 VK_DYNAMIC_STATE_DEPTH_BIAS};
    auto dy = create_info<VkPipelineDynamicStateCreateInfo>();
    dy.dynamicStateCount = static_cast<u32>(dynamic_states.size());
    dy.pDynamicStates = dynamic_states.data();

    std::array<VkVertexInputBindingDescription, 1> binding_descriptions{VkVertexInputBindingDescription{
            .binding = 0,
            .stride = sizeof(PositionOnlyVertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    }};

    std::array<VkVertexInputAttributeDescription, 2> attribute_descriptions{
            VkVertexInputAttributeDescription{
                    .location = 0,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(PositionOnlyVertex, pos),
            },
            VkVertexInputAttributeDescription{
                    .location = 1,
                    .binding = 0,
                    .format = VK_FORMAT_R32_UINT,
                    .offset = offsetof(PositionOnlyVertex, uvs),
            },
    };

    auto vertex_input = create_info<VkPipelineVertexInputStateCreateInfo>();
    vertex_input.vertexBindingDescriptionCount = static_cast<u32>(binding_descriptions.size());
    vertex_input.pVertexBindingDescriptions = binding_descriptions.data();
    vertex_input.vertexAttributeDescriptionCount = static_cast<u32>(attribute_descriptions.size());
    vertex_input.pVertexAttributeDescriptions = attribute_descriptions.data();

    auto assembly_state = create_info<VkPipelineInputAssemblyStateCreateInfo>();
    assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    assembly_state.primitiveRestartEnable = VK_FALSE;

    auto ci = create_info<VkGraphicsPipelineCreateInfo>();
    ci.pNext = &rendering_info;
    ci.stageCount = static_cast<u32>(stages.size());
    ci.pStages = stages.data();
    ci.pVertexInputState = &vertex_input;
    ci.pInputAssemblyState = &assembly_state;
    ci.pViewportState = &vp;
    ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms;
    ci.pDepthStencilState = &ds;
    ci.pColorBlendState = &cb;
    ci.pDynamicState = &dy;
    ci.layout = layout;
    ci.basePipelineHandle = VK_NULL_HANDLE;
    ci.basePipelineIndex = -1;

    VkPipeline pipeline;
    VkPipelineCache cache_handle = cache ? cache->get() : VK_NULL_HANDLE;
    vkCreateGraphicsPipelines(device, cache_handle, 1, &ci, nullptr, &pipeline);
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, "directional_shadow_map");

    vkDestroyShaderModule(device, vert_module, nullptr);
    vkDestroyShaderModule(device, frag_module, nullptr);

    return {pipeline, layout};
}
auto create_directional_shadow_map_pipeline(VkDevice device, PipelineCache *cache,
                                            VkDescriptorSetLayout bindless_layout, const std::vector<u32> &vert_code,
                                            VkFormat depth_format, VkSampleCountFlagBits samples) -> CompiledPipeline {
    VkShaderModule vert_module{};
    auto shader_ci = create_info<VkShaderModuleCreateInfo>();
    shader_ci.codeSize = vert_code.size() * sizeof(u32);
    shader_ci.pCode = vert_code.data();
    vk_check(vkCreateShaderModule(device, &shader_ci, nullptr, &vert_module));

    auto vert_stage_ci = create_info<VkPipelineShaderStageCreateInfo>();
    vert_stage_ci.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_stage_ci.module = vert_module;
    vert_stage_ci.pName = "shadow_vs_mdi";

    std::array stages = {
            vert_stage_ci,
    };

    // Pipeline Layout
    VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(ShadowMapPushConstants),
    };

    auto layout_ci = create_info<VkPipelineLayoutCreateInfo>();
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &bindless_layout;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &push_range;

    VkPipelineLayout layout;
    vkCreatePipelineLayout(device, &layout_ci, nullptr, &layout);
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, layout, "shadow_map");

    // Depth State
    auto ds = create_info<VkPipelineDepthStencilStateCreateInfo>();
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL; // Reverse-Z
    ds.minDepthBounds = 0.0f;
    ds.maxDepthBounds = 1.0f;

    // No Color Attachments (depth-only pass)
    auto cb = create_info<VkPipelineColorBlendStateCreateInfo>();
    cb.attachmentCount = 0;
    cb.pAttachments = nullptr;

    // Rasterization with depth bias for shadow acne
    auto rs = create_info<VkPipelineRasterizationStateCreateInfo>();
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    rs.depthBiasEnable = VK_TRUE;
    rs.depthBiasConstantFactor = 0.0f; // Set dynamically
    rs.depthBiasClamp = 0.0f;
    rs.depthBiasSlopeFactor = 0.0f; // Set dynamically

    // Dynamic Rendering Info
    auto rendering_info = create_info<VkPipelineRenderingCreateInfo>();
    rendering_info.depthAttachmentFormat = depth_format;

    auto vp = create_info<VkPipelineViewportStateCreateInfo>();
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    auto ms = create_info<VkPipelineMultisampleStateCreateInfo>();
    ms.rasterizationSamples = samples;

    std::array dynamic_states = {VK_DYNAMIC_STATE_VIEWPORT,         VK_DYNAMIC_STATE_SCISSOR,
                                 VK_DYNAMIC_STATE_DEPTH_COMPARE_OP, VK_DYNAMIC_STATE_DEPTH_BOUNDS,
                                 VK_DYNAMIC_STATE_CULL_MODE,        VK_DYNAMIC_STATE_FRONT_FACE,
                                 VK_DYNAMIC_STATE_DEPTH_BIAS};
    auto dy = create_info<VkPipelineDynamicStateCreateInfo>();
    dy.dynamicStateCount = static_cast<u32>(dynamic_states.size());
    dy.pDynamicStates = dynamic_states.data();

    std::array<VkVertexInputBindingDescription, 1> binding_descriptions{VkVertexInputBindingDescription{
            .binding = 0,
            .stride = sizeof(PositionOnlyVertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    }};

    std::array<VkVertexInputAttributeDescription, 2> attribute_descriptions{
            VkVertexInputAttributeDescription{
                    .location = 0,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(PositionOnlyVertex, pos),
            },
            VkVertexInputAttributeDescription{
                    .location = 1,
                    .binding = 0,
                    .format = VK_FORMAT_R32_UINT,
                    .offset = offsetof(PositionOnlyVertex, uvs),
            },
    };

    auto vertex_input = create_info<VkPipelineVertexInputStateCreateInfo>();
    vertex_input.vertexBindingDescriptionCount = static_cast<u32>(binding_descriptions.size());
    vertex_input.pVertexBindingDescriptions = binding_descriptions.data();
    vertex_input.vertexAttributeDescriptionCount = static_cast<u32>(attribute_descriptions.size());
    vertex_input.pVertexAttributeDescriptions = attribute_descriptions.data();

    auto assembly_state = create_info<VkPipelineInputAssemblyStateCreateInfo>();
    assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    assembly_state.primitiveRestartEnable = VK_FALSE;

    auto ci = create_info<VkGraphicsPipelineCreateInfo>();
    ci.pNext = &rendering_info;
    ci.stageCount = static_cast<u32>(stages.size());
    ci.pStages = stages.data();
    ci.pVertexInputState = &vertex_input;
    ci.pInputAssemblyState = &assembly_state;
    ci.pViewportState = &vp;
    ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms;
    ci.pDepthStencilState = &ds;
    ci.pColorBlendState = &cb;
    ci.pDynamicState = &dy;
    ci.layout = layout;
    ci.basePipelineHandle = VK_NULL_HANDLE;
    ci.basePipelineIndex = -1;

    VkPipeline pipeline;
    VkPipelineCache cache_handle = cache ? cache->get() : VK_NULL_HANDLE;

    vkCreateGraphicsPipelines(device, cache_handle, 1, &ci, nullptr, &pipeline);
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, "directional_shadow_map");

    vkDestroyShaderModule(device, vert_module, nullptr);

    return {.pipeline = pipeline, .layout = layout};
}


auto create_tonemap_pipeline(VkDevice device, PipelineCache *cache, VkDescriptorSetLayout layout,
                             const std::vector<u32> &vert_code, const std::vector<u32> &frag_code,
                             const std::string_view vert_entry, const std::string_view frag_entry,
                             VkFormat color_format) -> CompiledPipeline {
    VkShaderModule vert_shader{};
    auto vert_create_info = create_info<VkShaderModuleCreateInfo>();
    vert_create_info.codeSize = vert_code.size() * sizeof(u32);
    vert_create_info.pCode = vert_code.data();
    vk_check(vkCreateShaderModule(device, &vert_create_info, nullptr, &vert_shader));

    VkShaderModule frag_shader{};
    auto frag_create_info = create_info<VkShaderModuleCreateInfo>();
    frag_create_info.codeSize = frag_code.size() * sizeof(u32);
    frag_create_info.pCode = frag_code.data();
    vk_check(vkCreateShaderModule(device, &frag_create_info, nullptr, &frag_shader));

    VkPushConstantRange push_constant_range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(TonemapPushConstants),
    };
    VkPipelineLayout pipeline_layout{};
    auto plci = create_info<VkPipelineLayoutCreateInfo>();
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &push_constant_range;
    vk_check(vkCreatePipelineLayout(device, &plci, nullptr, &pipeline_layout));
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, pipeline_layout, "tonemap");


    auto vert_stage_ci = create_info<VkPipelineShaderStageCreateInfo>();
    vert_stage_ci.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_stage_ci.module = vert_shader;
    vert_stage_ci.pName = vert_entry.data();

    auto frag_stage_ci = create_info<VkPipelineShaderStageCreateInfo>();
    frag_stage_ci.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_stage_ci.module = frag_shader;
    frag_stage_ci.pName = frag_entry.data();

    std::array shader_stages{vert_stage_ci, frag_stage_ci};

    auto vertex_input = create_info<VkPipelineVertexInputStateCreateInfo>();
    vertex_input.vertexBindingDescriptionCount = 0;
    vertex_input.pVertexBindingDescriptions = nullptr;
    vertex_input.vertexAttributeDescriptionCount = 0;
    vertex_input.pVertexAttributeDescriptions = nullptr;

    auto input_assembly = create_info<VkPipelineInputAssemblyStateCreateInfo>();
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly.primitiveRestartEnable = VK_FALSE;

    auto viewport_state = create_info<VkPipelineViewportStateCreateInfo>();
    viewport_state.viewportCount = 1;
    viewport_state.pViewports = nullptr; // dynamic
    viewport_state.scissorCount = 1;
    viewport_state.pScissors = nullptr; // dynamic

    auto rasterization = create_info<VkPipelineRasterizationStateCreateInfo>();
    rasterization.depthClampEnable = VK_FALSE;
    rasterization.rasterizerDiscardEnable = VK_FALSE;
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization.depthBiasEnable = VK_FALSE;
    rasterization.depthBiasConstantFactor = 0.0f;
    rasterization.depthBiasClamp = 0.0f;
    rasterization.depthBiasSlopeFactor = 0.0f;
    rasterization.lineWidth = 1.0f;

    auto multisample = create_info<VkPipelineMultisampleStateCreateInfo>();
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisample.sampleShadingEnable = VK_FALSE;
    multisample.minSampleShading = 1.0f;
    multisample.pSampleMask = nullptr;
    multisample.alphaToCoverageEnable = VK_FALSE;
    multisample.alphaToOneEnable = VK_FALSE;

    auto depth_stencil = create_info<VkPipelineDepthStencilStateCreateInfo>();
    depth_stencil.depthTestEnable = VK_FALSE;
    depth_stencil.depthWriteEnable = VK_FALSE;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    depth_stencil.depthBoundsTestEnable = VK_FALSE;
    depth_stencil.stencilTestEnable = VK_FALSE;
    depth_stencil.minDepthBounds = 1.0f;
    depth_stencil.maxDepthBounds = 0.0f;

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

    auto color_blend = create_info<VkPipelineColorBlendStateCreateInfo>();
    color_blend.logicOpEnable = VK_FALSE;
    color_blend.logicOp = VK_LOGIC_OP_COPY;
    color_blend.attachmentCount = 1;
    color_blend.pAttachments = &color_blend_attachment;
    color_blend.blendConstants[0] = 0.0f;
    color_blend.blendConstants[1] = 0.0f;
    color_blend.blendConstants[2] = 0.0f;
    color_blend.blendConstants[3] = 0.0f;

    std::array dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    auto dynamic_state = create_info<VkPipelineDynamicStateCreateInfo>();
    dynamic_state.dynamicStateCount = static_cast<u32>(dynamic_states.size());
    dynamic_state.pDynamicStates = dynamic_states.data();

    auto rendering_info = create_info<VkPipelineRenderingCreateInfo>();
    rendering_info.viewMask = 0;
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachmentFormats = &color_format;
    rendering_info.depthAttachmentFormat = VK_FORMAT_UNDEFINED;
    rendering_info.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

    auto pipeline_info = create_info<VkGraphicsPipelineCreateInfo>();
    pipeline_info.pNext = &rendering_info;
    pipeline_info.stageCount = static_cast<u32>(shader_stages.size());
    pipeline_info.pStages = shader_stages.data();
    pipeline_info.pVertexInputState = &vertex_input;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterization;
    pipeline_info.pMultisampleState = &multisample;
    pipeline_info.pDepthStencilState = &depth_stencil;
    pipeline_info.pColorBlendState = &color_blend;
    pipeline_info.pDynamicState = &dynamic_state;
    pipeline_info.layout = pipeline_layout;
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
    pipeline_info.basePipelineIndex = -1;

    VkPipeline pipeline{};
    VkPipelineCache cache_handle = cache ? cache->get() : VK_NULL_HANDLE;
    vk_check(vkCreateGraphicsPipelines(device, cache_handle, 1, &pipeline_info, nullptr, &pipeline));
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, "tonemap");

    vkDestroyShaderModule(device, vert_shader, nullptr);
    vkDestroyShaderModule(device, frag_shader, nullptr);

    return CompiledPipeline{pipeline, pipeline_layout};
}

auto create_gbuffer_pipeline(VkDevice device, PipelineCache *cache, VkDescriptorSetLayout bindless_layout,
                             const std::vector<u32> &vert_code, const std::vector<u32> &frag_code,
                             VkFormat gbuffer0_format, VkFormat gbuffer1_format, VkFormat gbuffer2_format,
                             VkFormat depth_format) -> CompiledPipeline {
    VkShaderModule vert_module{};
    {
        auto ci = create_info<VkShaderModuleCreateInfo>();
        ci.codeSize = vert_code.size() * sizeof(u32);
        ci.pCode = vert_code.data();
        vk_check(vkCreateShaderModule(device, &ci, nullptr, &vert_module));
    }

    VkShaderModule frag_module{};
    {
        auto ci = create_info<VkShaderModuleCreateInfo>();
        ci.codeSize = frag_code.size() * sizeof(u32);
        ci.pCode = frag_code.data();
        vk_check(vkCreateShaderModule(device, &ci, nullptr, &frag_module));
    }

    auto vert_stage_ci = create_info<VkPipelineShaderStageCreateInfo>();
    vert_stage_ci.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_stage_ci.module = vert_module;
    vert_stage_ci.pName = "main_vs_mdi";

    auto frag_stage_ci = create_info<VkPipelineShaderStageCreateInfo>();
    frag_stage_ci.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_stage_ci.module = frag_module;
    frag_stage_ci.pName = "main_fs_mdi";

    std::array stages{vert_stage_ci, frag_stage_ci};

    // Push constants for the GBuffer pass (your Slang GBufferPC)
    VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(RenderingPushConstants),
    };

    VkPipelineLayout pipeline_layout{};
    {
        auto plci = create_info<VkPipelineLayoutCreateInfo>();
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &bindless_layout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &push_range;
        vk_check(vkCreatePipelineLayout(device, &plci, nullptr, &pipeline_layout));
        set_debug_name(device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, pipeline_layout, "gbuffer_mrt_layout");
    }

    // Vertex input: position + normal + uvs + tangent + bitangent
    std::array<VkVertexInputBindingDescription, 1> bindings{
            VkVertexInputBindingDescription{
                    .binding = 0,
                    .stride = sizeof(Vertex),
                    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            },
    };

    std::array<VkVertexInputAttributeDescription, 5> attrs{
            VkVertexInputAttributeDescription{
                    .location = 0,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, position),
            },
            VkVertexInputAttributeDescription{
                    .location = 1,
                    .binding = 0,
                    .format = VK_FORMAT_R32_UINT,
                    .offset = offsetof(Vertex, uvs),
            },
            VkVertexInputAttributeDescription{
                    .location = 2,
                    .binding = 0,
                    .format = VK_FORMAT_R32_UINT,
                    .offset = offsetof(Vertex, normal),
            },
            VkVertexInputAttributeDescription{
                    .location = 3,
                    .binding = 0,
                    .format = VK_FORMAT_R32_UINT,
                    .offset = offsetof(Vertex, tangent),
            },
            VkVertexInputAttributeDescription{
                    .location = 4,
                    .binding = 0,
                    .format = VK_FORMAT_R32_UINT,
                    .offset = offsetof(Vertex, bitangent),
            },
    };

    auto vi = create_info<VkPipelineVertexInputStateCreateInfo>();
    vi.vertexBindingDescriptionCount = static_cast<u32>(bindings.size());
    vi.pVertexBindingDescriptions = bindings.data();
    vi.vertexAttributeDescriptionCount = static_cast<u32>(attrs.size());
    vi.pVertexAttributeDescriptions = attrs.data();

    auto ia = create_info<VkPipelineInputAssemblyStateCreateInfo>();
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    auto vp = create_info<VkPipelineViewportStateCreateInfo>();
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    auto rs = create_info<VkPipelineRasterizationStateCreateInfo>();
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    auto ms = create_info<VkPipelineMultisampleStateCreateInfo>();
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // You're doing predepth, so keep depth test ON, depth write OFF.
    // Compare op is dynamic in your engine, but set a sane default.
    auto ds = create_info<VkPipelineDepthStencilStateCreateInfo>();
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_FALSE;
    ds.depthCompareOp = VK_COMPARE_OP_EQUAL; // matches your vkCmdSetDepthCompareOp(EQUAL)
    ds.depthBoundsTestEnable = VK_FALSE;
    ds.stencilTestEnable = VK_FALSE;
    ds.minDepthBounds = 0.0f;
    ds.maxDepthBounds = 1.0f;

    // MRT: 3 attachments, no blending for gbuffer writes.
    VkPipelineColorBlendAttachmentState blend0{};
    blend0.blendEnable = VK_FALSE;
    blend0.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    std::array<VkPipelineColorBlendAttachmentState, 3> blends{blend0, blend0, blend0};

    auto cb = create_info<VkPipelineColorBlendStateCreateInfo>();
    cb.attachmentCount = static_cast<u32>(blends.size());
    cb.pAttachments = blends.data();

    std::array<VkDynamicState, 6> dyn_states{
            VK_DYNAMIC_STATE_VIEWPORT,     VK_DYNAMIC_STATE_SCISSOR,   VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
            VK_DYNAMIC_STATE_DEPTH_BOUNDS, VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE,
    };

    auto dy = create_info<VkPipelineDynamicStateCreateInfo>();
    dy.dynamicStateCount = static_cast<u32>(dyn_states.size());
    dy.pDynamicStates = dyn_states.data();

    std::array<VkFormat, 3> color_formats{gbuffer0_format, gbuffer1_format, gbuffer2_format};

    auto rendering_info = create_info<VkPipelineRenderingCreateInfo>();
    rendering_info.colorAttachmentCount = static_cast<u32>(color_formats.size());
    rendering_info.pColorAttachmentFormats = color_formats.data();
    rendering_info.depthAttachmentFormat = depth_format;
    rendering_info.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

    auto gpci = create_info<VkGraphicsPipelineCreateInfo>();
    gpci.pNext = &rendering_info;
    gpci.stageCount = static_cast<u32>(stages.size());
    gpci.pStages = stages.data();
    gpci.pVertexInputState = &vi;
    gpci.pInputAssemblyState = &ia;
    gpci.pViewportState = &vp;
    gpci.pRasterizationState = &rs;
    gpci.pMultisampleState = &ms;
    gpci.pDepthStencilState = &ds;
    gpci.pColorBlendState = &cb;
    gpci.pDynamicState = &dy;
    gpci.layout = pipeline_layout;
    gpci.basePipelineHandle = VK_NULL_HANDLE;
    gpci.basePipelineIndex = -1;

    VkPipeline pipeline{};
    VkPipelineCache cache_handle = cache ? cache->get() : VK_NULL_HANDLE;

    vk_check(vkCreateGraphicsPipelines(device, cache_handle, 1, &gpci, nullptr, &pipeline));
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, "gbuffer_mrt");

    vkDestroyShaderModule(device, vert_module, nullptr);
    vkDestroyShaderModule(device, frag_module, nullptr);

    return CompiledPipeline{pipeline, pipeline_layout};
}

auto create_deferred_lighting_graphics_pipeline(VkDevice device, PipelineCache *cache,
                                                VkDescriptorSetLayout bindless_layout,
                                                const std::vector<u32> &frag_code, const VkShaderModule vert,
                                                std::string_view frag_entry, VkFormat color_format)
        -> CompiledPipeline {
    VkShaderModule frag_shader{};
    {
        auto ci = create_info<VkShaderModuleCreateInfo>();
        ci.codeSize = frag_code.size() * sizeof(u32);
        ci.pCode = frag_code.data();
        vk_check(vkCreateShaderModule(device, &ci, nullptr, &frag_shader));
        set_debug_name(device, VK_OBJECT_TYPE_SHADER_MODULE, frag_shader, "deferred_lighting_vs");
    }

    VkPushConstantRange push_constant_range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(DeferredLightingPushConstants),
    };

    VkPipelineLayout pipeline_layout{};
    {
        auto plci = create_info<VkPipelineLayoutCreateInfo>();
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &bindless_layout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &push_constant_range;
        vk_check(vkCreatePipelineLayout(device, &plci, nullptr, &pipeline_layout));
        set_debug_name(device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, pipeline_layout, "deferred_lighting_layout");
    }

    auto vert_stage_ci = create_info<VkPipelineShaderStageCreateInfo>();
    vert_stage_ci.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_stage_ci.module = vert;
    vert_stage_ci.pName = Pipeline::Fullscreen::vs_entry.data();

    auto frag_stage_ci = create_info<VkPipelineShaderStageCreateInfo>();
    frag_stage_ci.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_stage_ci.module = frag_shader;
    frag_stage_ci.pName = frag_entry.data();

    std::array shader_stages{vert_stage_ci, frag_stage_ci};

    // Fullscreen triangle: no vertex buffers.
    auto vertex_input = create_info<VkPipelineVertexInputStateCreateInfo>();
    vertex_input.vertexBindingDescriptionCount = 0;
    vertex_input.pVertexBindingDescriptions = nullptr;
    vertex_input.vertexAttributeDescriptionCount = 0;
    vertex_input.pVertexAttributeDescriptions = nullptr;

    auto input_assembly = create_info<VkPipelineInputAssemblyStateCreateInfo>();
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly.primitiveRestartEnable = VK_FALSE;

    auto viewport_state = create_info<VkPipelineViewportStateCreateInfo>();
    viewport_state.viewportCount = 1;
    viewport_state.pViewports = nullptr; // dynamic
    viewport_state.scissorCount = 1;
    viewport_state.pScissors = nullptr; // dynamic

    // No culling for fullscreen triangle.
    auto rasterization = create_info<VkPipelineRasterizationStateCreateInfo>();
    rasterization.depthClampEnable = VK_FALSE;
    rasterization.rasterizerDiscardEnable = VK_FALSE;
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization.depthBiasEnable = VK_FALSE;
    rasterization.depthBiasConstantFactor = 0.0f;
    rasterization.depthBiasClamp = 0.0f;
    rasterization.depthBiasSlopeFactor = 0.0f;
    rasterization.lineWidth = 1.0f;

    auto multisample = create_info<VkPipelineMultisampleStateCreateInfo>();
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisample.sampleShadingEnable = VK_FALSE;
    multisample.minSampleShading = 1.0f;
    multisample.pSampleMask = nullptr;
    multisample.alphaToCoverageEnable = VK_FALSE;
    multisample.alphaToOneEnable = VK_FALSE;

    // Deferred lighting pass: no depth test/write.
    auto depth_stencil = create_info<VkPipelineDepthStencilStateCreateInfo>();
    depth_stencil.depthTestEnable = VK_FALSE;
    depth_stencil.depthWriteEnable = VK_FALSE;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    depth_stencil.depthBoundsTestEnable = VK_FALSE;
    depth_stencil.stencilTestEnable = VK_FALSE;
    depth_stencil.minDepthBounds = 0.0f;
    depth_stencil.maxDepthBounds = 1.0f;

    VkPipelineColorBlendAttachmentState color_blend_attachment{
            .blendEnable = VK_FALSE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                              VK_COLOR_COMPONENT_A_BIT,
    };

    auto color_blend = create_info<VkPipelineColorBlendStateCreateInfo>();
    color_blend.logicOpEnable = VK_FALSE;
    color_blend.logicOp = VK_LOGIC_OP_COPY;
    color_blend.attachmentCount = 1;
    color_blend.pAttachments = &color_blend_attachment;
    color_blend.blendConstants[0] = 0.0f;
    color_blend.blendConstants[1] = 0.0f;
    color_blend.blendConstants[2] = 0.0f;
    color_blend.blendConstants[3] = 0.0f;

    std::array dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    auto dynamic_state = create_info<VkPipelineDynamicStateCreateInfo>();
    dynamic_state.dynamicStateCount = static_cast<u32>(dynamic_states.size());
    dynamic_state.pDynamicStates = dynamic_states.data();

    auto rendering_info = create_info<VkPipelineRenderingCreateInfo>();
    rendering_info.viewMask = 0;
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachmentFormats = &color_format;
    rendering_info.depthAttachmentFormat = VK_FORMAT_UNDEFINED;
    rendering_info.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

    auto gpci = create_info<VkGraphicsPipelineCreateInfo>();
    gpci.pNext = &rendering_info;
    gpci.stageCount = static_cast<u32>(shader_stages.size());
    gpci.pStages = shader_stages.data();
    gpci.pVertexInputState = &vertex_input;
    gpci.pInputAssemblyState = &input_assembly;
    gpci.pViewportState = &viewport_state;
    gpci.pRasterizationState = &rasterization;
    gpci.pMultisampleState = &multisample;
    gpci.pDepthStencilState = &depth_stencil;
    gpci.pColorBlendState = &color_blend;
    gpci.pDynamicState = &dynamic_state;
    gpci.layout = pipeline_layout;
    gpci.basePipelineHandle = VK_NULL_HANDLE;
    gpci.basePipelineIndex = -1;

    VkPipeline pipeline{};
    VkPipelineCache cache_handle = cache ? cache->get() : VK_NULL_HANDLE;

    vk_check(vkCreateGraphicsPipelines(device, cache_handle, 1, &gpci, nullptr, &pipeline));
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, "deferred_lighting_fs");

    vkDestroyShaderModule(device, frag_shader, nullptr);

    return CompiledPipeline{.pipeline = pipeline, .layout = pipeline_layout};
}

namespace {
    static constexpr std::array<unsigned char, 860> fullscreen_vs_spv = {
            0x03, 0x02, 0x23, 0x07, 0x00, 0x04, 0x01, 0x00, 0x00, 0x00, 0x28, 0x00, 0x29, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x11, 0x00, 0x02, 0x00, 0x4b, 0x11, 0x00, 0x00, 0x11, 0x00, 0x02, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x0e, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x09, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x6d, 0x61, 0x69, 0x6e, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
            0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x05, 0x00,
            0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x48, 0x11, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x04, 0x00, 0x00, 0x00,
            0x0b, 0x00, 0x00, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x02, 0x00, 0x00, 0x00, 0x0b, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x03, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x13, 0x00, 0x02, 0x00, 0x06, 0x00, 0x00, 0x00, 0x21, 0x00, 0x03, 0x00, 0x07, 0x00,
            0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x16, 0x00, 0x03, 0x00, 0x08, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
            0x17, 0x00, 0x04, 0x00, 0x09, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x17, 0x00,
            0x04, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x15, 0x00, 0x04, 0x00,
            0x0b, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x0c, 0x00,
            0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x15, 0x00, 0x04, 0x00, 0x0d, 0x00, 0x00, 0x00,
            0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x14, 0x00, 0x02, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x2b, 0x00,
            0x04, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00,
            0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf, 0x2b, 0x00, 0x04, 0x00, 0x08, 0x00,
            0x00, 0x00, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x40, 0x2b, 0x00, 0x04, 0x00, 0x0d, 0x00, 0x00, 0x00,
            0x12, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x13, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x80, 0x3f, 0x2b, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x15, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x3f, 0x20, 0x00, 0x04, 0x00, 0x16, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
            0x20, 0x00, 0x04, 0x00, 0x17, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x3b, 0x00,
            0x04, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00,
            0x0c, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00, 0x16, 0x00,
            0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00, 0x17, 0x00, 0x00, 0x00,
            0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x05, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x18, 0x00,
            0x00, 0x00, 0x15, 0x00, 0x00, 0x00, 0x15, 0x00, 0x00, 0x00, 0x36, 0x00, 0x05, 0x00, 0x06, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x19, 0x00,
            0x00, 0x00, 0x3d, 0x00, 0x04, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
            0x3d, 0x00, 0x04, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x82, 0x00,
            0x05, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00,
            0x7c, 0x00, 0x04, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x1d, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0xaa, 0x00,
            0x05, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x1d, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00,
            0xa9, 0x00, 0x06, 0x00, 0x08, 0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x11, 0x00,
            0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0xaa, 0x00, 0x05, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
            0x1d, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00, 0xa9, 0x00, 0x06, 0x00, 0x08, 0x00, 0x00, 0x00, 0x21, 0x00,
            0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x50, 0x00, 0x05, 0x00,
            0x0a, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00, 0x50, 0x00,
            0x06, 0x00, 0x09, 0x00, 0x00, 0x00, 0x23, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00,
            0x14, 0x00, 0x00, 0x00, 0x8e, 0x00, 0x05, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x22, 0x00,
            0x00, 0x00, 0x15, 0x00, 0x00, 0x00, 0x81, 0x00, 0x05, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x25, 0x00, 0x00, 0x00,
            0x24, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00, 0x08, 0x00, 0x00, 0x00, 0x26, 0x00,
            0x00, 0x00, 0x25, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x83, 0x00, 0x05, 0x00, 0x08, 0x00, 0x00, 0x00,
            0x27, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x26, 0x00, 0x00, 0x00, 0x52, 0x00, 0x06, 0x00, 0x0a, 0x00,
            0x00, 0x00, 0x28, 0x00, 0x00, 0x00, 0x27, 0x00, 0x00, 0x00, 0x25, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x3e, 0x00, 0x03, 0x00, 0x02, 0x00, 0x00, 0x00, 0x23, 0x00, 0x00, 0x00, 0x3e, 0x00, 0x03, 0x00, 0x03, 0x00,
            0x00, 0x00, 0x28, 0x00, 0x00, 0x00, 0xfd, 0x00, 0x01, 0x00, 0x38, 0x00, 0x01, 0x00};
}

namespace Pipeline {

    auto get_or_create_fullscreen_vs(RenderContext &ctx) -> u32 {
        auto handle = ctx.shaders.maybe_get_handle(EngineShaderIndices::fullscreen_vertex_shader);
        if (handle.valid()) {
            return handle.index();
        }

        std::array<u32, sizeof(fullscreen_vs_spv) / sizeof(u32)> vs_code{};
        std::memcpy(vs_code.data(), fullscreen_vs_spv.data(), fullscreen_vs_spv.size());

        auto smci = create_info<VkShaderModuleCreateInfo>();
        smci.codeSize = vs_code.size() * sizeof(u32);
        smci.pCode = vs_code.data();

        VkShaderModule shader_module{};
        vk_check(vkCreateShaderModule(ctx.get_device(), &smci, nullptr, &shader_module));
        return ctx.create_shader(std::move(shader_module)).index();
    }


    auto create_fullscreen_pipeline(const Fullscreen &info) -> CompiledPipeline {
        VkShaderModule frag_shader{};
        {
            VkShaderModuleCreateInfo ci = create_info<VkShaderModuleCreateInfo>();
            ci.codeSize = info.frag_code.size() * sizeof(u32);
            ci.pCode = info.frag_code.data();
            vk_check(vkCreateShaderModule(info.device, &ci, nullptr, &frag_shader));
        }

        VkPushConstantRange push_range{
                .stageFlags = info.push_constant_stages,
                .offset = 0,
                .size = info.push_constant_size,
        };

        VkPipelineLayout pipeline_layout{};
        {
            auto plci = create_info<VkPipelineLayoutCreateInfo>();
            plci.setLayoutCount = 1;
            plci.pSetLayouts = &info.bindless_layout;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &push_range;
            vk_check(vkCreatePipelineLayout(info.device, &plci, nullptr, &pipeline_layout));
        }

        auto vs_stage = create_info<VkPipelineShaderStageCreateInfo>();
        vs_stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vs_stage.module = info.fullscreen_vs;
        vs_stage.pName = info.vs_entry.data();

        auto fs_stage = create_info<VkPipelineShaderStageCreateInfo>();
        fs_stage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fs_stage.module = frag_shader;
        fs_stage.pName = info.fs_entry.data();

        std::array stages{vs_stage, fs_stage};

        auto vi = create_info<VkPipelineVertexInputStateCreateInfo>();

        VkPipelineInputAssemblyStateCreateInfo ia = create_info<VkPipelineInputAssemblyStateCreateInfo>();
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp = create_info<VkPipelineViewportStateCreateInfo>();
        vp.viewportCount = 1;
        vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs = create_info<VkPipelineRasterizationStateCreateInfo>();
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms = create_info<VkPipelineMultisampleStateCreateInfo>();
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds = create_info<VkPipelineDepthStencilStateCreateInfo>();
        ds.depthTestEnable = VK_FALSE;
        ds.depthWriteEnable = VK_FALSE;
        ds.depthCompareOp = VK_COMPARE_OP_ALWAYS;

        VkPipelineColorBlendAttachmentState att{};
        att.blendEnable = info.enable_blend ? VK_TRUE : VK_FALSE;
        att.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        att.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        att.colorBlendOp = VK_BLEND_OP_ADD;
        att.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        att.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        att.alphaBlendOp = VK_BLEND_OP_ADD;
        att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                             VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo cb = create_info<VkPipelineColorBlendStateCreateInfo>();
        cb.attachmentCount = 1;
        cb.pAttachments = &att;

        std::array dyn_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dy = create_info<VkPipelineDynamicStateCreateInfo>();
        dy.dynamicStateCount = static_cast<u32>(dyn_states.size());
        dy.pDynamicStates = dyn_states.data();

        VkPipelineRenderingCreateInfo ri = create_info<VkPipelineRenderingCreateInfo>();
        ri.colorAttachmentCount = 1;
        ri.pColorAttachmentFormats = &info.color_format;

        VkGraphicsPipelineCreateInfo gpci = create_info<VkGraphicsPipelineCreateInfo>();
        gpci.pNext = &ri;
        gpci.stageCount = static_cast<u32>(stages.size());
        gpci.pStages = stages.data();
        gpci.pVertexInputState = &vi;
        gpci.pInputAssemblyState = &ia;
        gpci.pViewportState = &vp;
        gpci.pRasterizationState = &rs;
        gpci.pMultisampleState = &ms;
        gpci.pDepthStencilState = &ds;
        gpci.pColorBlendState = &cb;
        gpci.pDynamicState = &dy;
        gpci.layout = pipeline_layout;

        VkPipeline pipeline{};
        VkPipelineCache cache_handle = info.cache ? info.cache->get() : VK_NULL_HANDLE;
        vk_check(vkCreateGraphicsPipelines(info.device, cache_handle, 1, &gpci, nullptr, &pipeline));

        vkDestroyShaderModule(info.device, frag_shader, nullptr);

        return CompiledPipeline{.pipeline = pipeline, .layout = pipeline_layout};
    }

    auto create_graphics_pipeline(const Pipeline::Graphics &info) -> CompiledPipeline {
        std::vector<VkShaderModule> modules(info.stages.size());
        std::vector<VkPipelineShaderStageCreateInfo> stage_cis(info.stages.size());

        for (auto &&[i, v]: info.stages | std::views::enumerate) {
            auto smci = create_info<VkShaderModuleCreateInfo>();
            smci.codeSize = v.code.size() * sizeof(u32);
            smci.pCode = v.code.data();
            vk_check(vkCreateShaderModule(info.device, &smci, nullptr, &modules[i]));

            auto &s = stage_cis[i];
            s = create_info<VkPipelineShaderStageCreateInfo>();
            s.stage = v.stage;
            s.module = modules[i];
            s.pName = v.entry.data();
        }

        // --- Pipeline layout ---
        VkPushConstantRange push_range{
                .stageFlags = info.push_constant_stages,
                .offset = 0,
                .size = info.push_constant_size,
        };

        auto plci = create_info<VkPipelineLayoutCreateInfo>();
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &info.bindless_layout;
        if (info.push_constant_size > 0) {
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &push_range;
        }

        VkPipelineLayout layout{};
        vk_check(vkCreatePipelineLayout(info.device, &plci, nullptr, &layout));
        set_debug_name(info.device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, layout, info.debug_name);

        // --- Vertex input ---
        auto vi = create_info<VkPipelineVertexInputStateCreateInfo>();
        if (info.vertex_input) {
            vi.vertexBindingDescriptionCount = static_cast<u32>(info.vertex_input->bindings.size());
            vi.pVertexBindingDescriptions = info.vertex_input->bindings.data();
            vi.vertexAttributeDescriptionCount = static_cast<u32>(info.vertex_input->attributes.size());
            vi.pVertexAttributeDescriptions = info.vertex_input->attributes.data();
        }

        auto ia = create_info<VkPipelineInputAssemblyStateCreateInfo>();
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        // --- Viewport/Scissor ---
        auto vp = create_info<VkPipelineViewportStateCreateInfo>();
        vp.viewportCount = 1;
        vp.scissorCount = 1;

        // --- Rasterization ---
        auto rs = create_info<VkPipelineRasterizationStateCreateInfo>();
        rs.lineWidth = 1.0f;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.cullMode = [&] {
            switch (info.cull_mode) {
                case CullMode::back:
                    return VK_CULL_MODE_BACK_BIT;
                case CullMode::front:
                    return VK_CULL_MODE_FRONT_BIT;
                default:
                    return VK_CULL_MODE_NONE;
            }
        }();
        rs.depthBiasEnable = info.depth_bias ? VK_TRUE : VK_FALSE;

        // --- Multisample ---
        auto ms = create_info<VkPipelineMultisampleStateCreateInfo>();
        ms.rasterizationSamples = info.samples;

        // --- Depth/Stencil ---
        auto ds = create_info<VkPipelineDepthStencilStateCreateInfo>();
        switch (info.depth_mode) {
            case DepthMode::write:
                ds.depthTestEnable = VK_TRUE;
                ds.depthWriteEnable = VK_TRUE;
                ds.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
                break;
            case DepthMode::test_equal:
                ds.depthTestEnable = VK_TRUE;
                ds.depthWriteEnable = VK_FALSE;
                ds.depthCompareOp = VK_COMPARE_OP_EQUAL;
                break;
            case DepthMode::test_greater_equal:
                ds.depthTestEnable = VK_TRUE;
                ds.depthWriteEnable = VK_FALSE;
                ds.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
                break;
            case DepthMode::none:
            default:
                ds.depthTestEnable = VK_FALSE;
                ds.depthWriteEnable = VK_FALSE;
                ds.depthCompareOp = VK_COMPARE_OP_ALWAYS;
                break;
        }
        ds.minDepthBounds = 0.0f;
        ds.maxDepthBounds = 1.0f;

        // --- Color blend ---
        std::vector<VkPipelineColorBlendAttachmentState> blend_attachments;
        std::vector<VkFormat> color_formats;

        for (auto &att: info.color_attachments) {
            color_formats.push_back(att.format);
            VkPipelineColorBlendAttachmentState a{};
            a.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                               VK_COLOR_COMPONENT_A_BIT;
            if (att.blend_additive) {
                a.blendEnable = VK_TRUE;
                a.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
                a.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
                a.colorBlendOp = VK_BLEND_OP_ADD;
                a.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
                a.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                a.alphaBlendOp = VK_BLEND_OP_ADD;
            }
            blend_attachments.push_back(a);
        }

        auto cb = create_info<VkPipelineColorBlendStateCreateInfo>();
        cb.attachmentCount = static_cast<u32>(blend_attachments.size());
        cb.pAttachments = blend_attachments.empty() ? nullptr : blend_attachments.data();

        // --- Dynamic rendering ---
        auto ri = create_info<VkPipelineRenderingCreateInfo>();
        ri.colorAttachmentCount = static_cast<u32>(color_formats.size());
        ri.pColorAttachmentFormats = color_formats.empty() ? nullptr : color_formats.data();
        ri.depthAttachmentFormat = info.depth_format;
        ri.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

        // --- Dynamic states ---
        std::vector<VkDynamicState> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        for (auto s: info.extra_dynamic_states)
            dynamic_states.push_back(s);

        auto dy = create_info<VkPipelineDynamicStateCreateInfo>();
        dy.dynamicStateCount = static_cast<u32>(dynamic_states.size());
        dy.pDynamicStates = dynamic_states.data();

        // --- Final create ---
        auto gpci = create_info<VkGraphicsPipelineCreateInfo>();
        gpci.pNext = &ri;
        gpci.stageCount = static_cast<u32>(stage_cis.size());
        gpci.pStages = stage_cis.data();
        gpci.pVertexInputState = info.vertex_input ? &vi : nullptr;
        gpci.pInputAssemblyState = info.vertex_input ? &ia : nullptr;
        gpci.pViewportState = &vp;
        gpci.pRasterizationState = &rs;
        gpci.pMultisampleState = &ms;
        gpci.pDepthStencilState = &ds;
        gpci.pColorBlendState = &cb;
        gpci.pDynamicState = &dy;
        gpci.layout = layout;
        gpci.basePipelineHandle = VK_NULL_HANDLE;
        gpci.basePipelineIndex = -1;

        VkPipeline pipeline{};
        VkPipelineCache cache_handle = info.cache ? info.cache->get() : VK_NULL_HANDLE;
        vk_check(vkCreateGraphicsPipelines(info.device, cache_handle, 1, &gpci, nullptr, &pipeline));
        set_debug_name(info.device, VK_OBJECT_TYPE_PIPELINE, pipeline, info.debug_name);

        for (auto mod: modules)
            vkDestroyShaderModule(info.device, mod, nullptr);

        return {pipeline, layout};
    }
} // namespace Pipeline


auto create_light_volume_mesh_pipeline(VkDevice device, PipelineCache *cache, VkDescriptorSetLayout bindless_layout,
                                       const std::vector<u32> &task_code, const std::vector<u32> &mesh_code,
                                       const std::vector<u32> &frag_code, VkFormat color_format, VkFormat depth_format,
                                       VkSampleCountFlagBits samples) -> CompiledPipeline {
    VkShaderModule task_module{};
    auto task_ci = create_info<VkShaderModuleCreateInfo>();
    task_ci.codeSize = task_code.size() * sizeof(u32);
    task_ci.pCode = task_code.data();
    vk_check(vkCreateShaderModule(device, &task_ci, nullptr, &task_module));

    // 1. Create Shader Modules
    VkShaderModule mesh_module{};
    auto mesh_ci = create_info<VkShaderModuleCreateInfo>();
    mesh_ci.codeSize = mesh_code.size() * sizeof(u32);
    mesh_ci.pCode = mesh_code.data();
    vk_check(vkCreateShaderModule(device, &mesh_ci, nullptr, &mesh_module));

    VkShaderModule frag_module{};
    auto frag_ci = create_info<VkShaderModuleCreateInfo>();
    frag_ci.codeSize = frag_code.size() * sizeof(u32);
    frag_ci.pCode = frag_code.data();
    vk_check(vkCreateShaderModule(device, &frag_ci, nullptr, &frag_module));

    // 2. Stages
    auto task_stage_ci = create_info<VkPipelineShaderStageCreateInfo>();
    task_stage_ci.stage = VK_SHADER_STAGE_TASK_BIT_EXT;
    task_stage_ci.module = task_module;
    task_stage_ci.pName = "main_as"; // Entry point in Slang

    auto mesh_stage_ci = create_info<VkPipelineShaderStageCreateInfo>();
    mesh_stage_ci.stage = VK_SHADER_STAGE_MESH_BIT_EXT;
    mesh_stage_ci.module = mesh_module;
    mesh_stage_ci.pName = "main_ms"; // Entry point in Slang

    auto frag_stage_ci = create_info<VkPipelineShaderStageCreateInfo>();
    frag_stage_ci.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_stage_ci.module = frag_module;
    frag_stage_ci.pName = "main_fs_debug";

    std::array stages = {task_stage_ci, mesh_stage_ci, frag_stage_ci};

    VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(DebugClusteredPushConstants), //
    };

    auto layout_ci = create_info<VkPipelineLayoutCreateInfo>();
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &bindless_layout;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &push_range;

    VkPipelineLayout layout;
    vkCreatePipelineLayout(device, &layout_ci, nullptr, &layout);

    // 4. Color Blending (Additive for Light Volumes)
    VkPipelineColorBlendAttachmentState blend_attachment{
            .blendEnable = VK_TRUE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                              VK_COLOR_COMPONENT_A_BIT,
    };

    auto cb = create_info<VkPipelineColorBlendStateCreateInfo>();
    cb.attachmentCount = 1;
    cb.pAttachments = &blend_attachment;

    // 5. Depth State (Test against pre-depth, do NOT write)
    auto ds = create_info<VkPipelineDepthStencilStateCreateInfo>();
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_FALSE;
    ds.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL; // Reverse-Z

    // 6. Rasterization (Front-Face culling so we see volume when inside)
    auto rs = create_info<VkPipelineRasterizationStateCreateInfo>();
    rs.cullMode = VK_CULL_MODE_FRONT_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    // 7. Dynamic Rendering & Multisample
    auto rendering_info = create_info<VkPipelineRenderingCreateInfo>();
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachmentFormats = &color_format;
    rendering_info.depthAttachmentFormat = depth_format;

    auto ms = create_info<VkPipelineMultisampleStateCreateInfo>();
    ms.rasterizationSamples = samples;

    auto vp = create_info<VkPipelineViewportStateCreateInfo>();
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    std::array dynamic_states = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    auto dy = create_info<VkPipelineDynamicStateCreateInfo>();
    dy.dynamicStateCount = static_cast<u32>(dynamic_states.size());
    dy.pDynamicStates = dynamic_states.data();

    // 8. Final Pipeline Create (Note: no VertexInput or InputAssembly)
    auto ci = create_info<VkGraphicsPipelineCreateInfo>();
    ci.pNext = &rendering_info;
    ci.stageCount = static_cast<u32>(stages.size());
    ci.pStages = stages.data();
    ci.pViewportState = &vp;
    ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms;
    ci.pDepthStencilState = &ds;
    ci.pColorBlendState = &cb;
    ci.pDynamicState = &dy;
    ci.layout = layout;

    VkPipeline pipeline;
    VkPipelineCache cache_handle = cache ? cache->get() : VK_NULL_HANDLE;
    vkCreateGraphicsPipelines(device, cache_handle, 1, &ci, nullptr, &pipeline);

    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, "light_volume_mesh_pass");

    vkDestroyShaderModule(device, task_module, nullptr);
    vkDestroyShaderModule(device, mesh_module, nullptr);
    vkDestroyShaderModule(device, frag_module, nullptr);

    return {pipeline, layout};
}

auto Pipeline::create_mesh_pipeline(const Mesh &info) -> CompiledPipeline {
    std::vector<VkPipelineShaderStageCreateInfo> stage_cis;
    std::vector<VkShaderModule> modules;

    auto make_stage = [&](const ShaderStageInfo &s, VkShaderStageFlagBits stage_flag) {
        VkShaderModule mod{};
        auto smci = create_info<VkShaderModuleCreateInfo>();
        smci.codeSize = s.code.size() * sizeof(u32);
        smci.pCode = s.code.data();
        vk_check(vkCreateShaderModule(info.device, &smci, nullptr, &mod));
        modules.push_back(mod);

        auto ci = create_info<VkPipelineShaderStageCreateInfo>();
        ci.stage = stage_flag;
        ci.module = mod;
        ci.pName = s.entry.data();
        stage_cis.push_back(ci);
    };

    if (info.stages.task)
        make_stage(*info.stages.task, VK_SHADER_STAGE_TASK_BIT_EXT);
    make_stage(info.stages.mesh, VK_SHADER_STAGE_MESH_BIT_EXT);
    make_stage(info.stages.fragment, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPushConstantRange push_range{
            .stageFlags = info.push_constant_stages,
            .offset = 0,
            .size = info.push_constant_size,
    };

    auto plci = create_info<VkPipelineLayoutCreateInfo>();
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &info.bindless_layout;
    if (info.push_constant_size > 0) {
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &push_range;
    }

    VkPipelineLayout layout{};
    vk_check(vkCreatePipelineLayout(info.device, &plci, nullptr, &layout));
    set_debug_name(info.device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, layout, info.debug_name);

    auto rs = create_info<VkPipelineRasterizationStateCreateInfo>();
    rs.lineWidth = 1.0f;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.cullMode = [&] {
        switch (info.cull_mode) {
            case CullMode::back:
                return VK_CULL_MODE_BACK_BIT;
            case CullMode::front:
                return VK_CULL_MODE_FRONT_BIT;
            default:
                return VK_CULL_MODE_NONE;
        }
    }();
    rs.depthBiasEnable = info.depth_bias ? VK_TRUE : VK_FALSE;

    auto ms = create_info<VkPipelineMultisampleStateCreateInfo>();
    ms.rasterizationSamples = info.samples;

    auto ds = create_info<VkPipelineDepthStencilStateCreateInfo>();
    ds.minDepthBounds = 0.0f;
    ds.maxDepthBounds = 1.0f;
    switch (info.depth_mode) {
        case DepthMode::write:
            ds.depthTestEnable = VK_TRUE;
            ds.depthWriteEnable = VK_TRUE;
            ds.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
            break;
        case DepthMode::test_equal:
            ds.depthTestEnable = VK_TRUE;
            ds.depthWriteEnable = VK_FALSE;
            ds.depthCompareOp = VK_COMPARE_OP_EQUAL;
            break;
        case DepthMode::test_greater_equal:
            ds.depthTestEnable = VK_TRUE;
            ds.depthWriteEnable = VK_FALSE;
            ds.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
            break;
        case DepthMode::none:
        default:
            ds.depthTestEnable = VK_FALSE;
            ds.depthWriteEnable = VK_FALSE;
            ds.depthCompareOp = VK_COMPARE_OP_ALWAYS;
            break;
    }

    std::vector<VkPipelineColorBlendAttachmentState> blend_attachments;
    std::vector<VkFormat> color_formats;
    for (const auto &att: info.color_attachments) {
        color_formats.push_back(att.format);
        VkPipelineColorBlendAttachmentState a{};
        a.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                           VK_COLOR_COMPONENT_A_BIT;
        if (att.blend_additive) {
            a.blendEnable = VK_TRUE;
            a.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            a.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            a.colorBlendOp = VK_BLEND_OP_ADD;
            a.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            a.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            a.alphaBlendOp = VK_BLEND_OP_ADD;
        }
        blend_attachments.push_back(a);
    }

    auto cb = create_info<VkPipelineColorBlendStateCreateInfo>();
    cb.attachmentCount = static_cast<u32>(blend_attachments.size());
    cb.pAttachments = blend_attachments.empty() ? nullptr : blend_attachments.data();

    auto ri = create_info<VkPipelineRenderingCreateInfo>();
    ri.colorAttachmentCount = static_cast<u32>(color_formats.size());
    ri.pColorAttachmentFormats = color_formats.empty() ? nullptr : color_formats.data();
    ri.depthAttachmentFormat = info.depth_format;
    ri.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

    auto vp = create_info<VkPipelineViewportStateCreateInfo>();
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    std::vector<VkDynamicState> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT,           VK_DYNAMIC_STATE_SCISSOR,
                                               VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
                                               VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,   VK_DYNAMIC_STATE_CULL_MODE};
    for (auto s: info.extra_dynamic_states)
        dynamic_states.push_back(s);

    auto dy = create_info<VkPipelineDynamicStateCreateInfo>();
    dy.dynamicStateCount = static_cast<u32>(dynamic_states.size());
    dy.pDynamicStates = dynamic_states.data();

    auto gpci = create_info<VkGraphicsPipelineCreateInfo>();
    gpci.pNext = &ri;
    gpci.stageCount = static_cast<u32>(stage_cis.size());
    gpci.pStages = stage_cis.data();
    gpci.pViewportState = &vp;
    gpci.pRasterizationState = &rs;
    gpci.pMultisampleState = &ms;
    gpci.pDepthStencilState = &ds;
    gpci.pColorBlendState = &cb;
    gpci.pDynamicState = &dy;
    gpci.layout = layout;
    gpci.basePipelineHandle = VK_NULL_HANDLE;
    gpci.basePipelineIndex = -1;

    VkPipeline pipeline{};
    VkPipelineCache cache_handle = info.cache ? info.cache->get() : VK_NULL_HANDLE;
    vk_check(vkCreateGraphicsPipelines(info.device, cache_handle, 1, &gpci, nullptr, &pipeline));
    set_debug_name(info.device, VK_OBJECT_TYPE_PIPELINE, pipeline, info.debug_name);

    for (auto mod: modules)
        vkDestroyShaderModule(info.device, mod, nullptr);

    return {pipeline, layout};
}
