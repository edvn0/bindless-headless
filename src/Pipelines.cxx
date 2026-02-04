#include "Pipelines.hxx"
#include "BindlessHeadless.hxx"
#include "Mesh.hxx"
#include "PipelineCache.hxx"

#include <glm/glm.hpp>

auto create_compute_pipeline(VkDevice device, PipelineCache &cache, VkDescriptorSetLayout layout,
                             const std::vector<u32> &code, const std::string_view entry_name) -> CompiledPipeline {
    VkShaderModule compute_shader{};
    auto ci = create_info<VkShaderModuleCreateInfo>();
    ci.codeSize = code.size() * sizeof(u32);
    ci.pCode = code.data();
    vk_check(vkCreateShaderModule(device, &ci, nullptr, &compute_shader));

    VkPushConstantRange push_constant_range{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(PointLightCullingPushConstants),
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
    vk_check(vkCreateComputePipelines(device, cache, 1, &cpci, nullptr, &pipeline));
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, entry_name);

    vkDestroyShaderModule(device, compute_shader, nullptr);
    return {pipeline, pi_layout};
}

auto create_predepth_pipeline(VkDevice device, PipelineCache &cache, VkDescriptorSetLayout bindless_layout,
                              const std::vector<uint32_t> &vert_code, VkFormat depth_format,
                              VkSampleCountFlagBits samples) -> CompiledPipeline {
    VkShaderModule vert_module{};
    auto shader_ci = create_info<VkShaderModuleCreateInfo>();
    shader_ci.codeSize = vert_code.size() * sizeof(u32);
    shader_ci.pCode = vert_code.data();
    vk_check(vkCreateShaderModule(device, &shader_ci, nullptr, &vert_module));

    auto vert_stage_ci = create_info<VkPipelineShaderStageCreateInfo>();
    vert_stage_ci.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_stage_ci.module = vert_module;
    vert_stage_ci.pName = "main_vs_mdi";


        std::array stages = {vert_stage_ci ,};

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

        std::array<VkVertexInputBindingDescription, 1> binding_descriptions{
                VkVertexInputBindingDescription{
                        .binding = 0,
                        .stride = sizeof(glm::vec3) + sizeof(u32),
                        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
                }
        };

        std::array<VkVertexInputAttributeDescription, 2> attribute_descriptions{
                VkVertexInputAttributeDescription{
                        .location = 0,
                        .binding = 0,
                        .format = VK_FORMAT_R32G32B32_SFLOAT,
                        .offset = 0,
                },
                VkVertexInputAttributeDescription{
                        .location = 1,
                        .binding = 0,
                        .format = VK_FORMAT_R32_UINT,
                        .offset = sizeof(glm::vec3),
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
    ci.stageCount = static_cast<uint32_t>(stages.size());
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
    vkCreateGraphicsPipelines(device, cache, 1, &ci, nullptr, &pipeline);
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, "predepth");

    // Cleanup local modules
    vkDestroyShaderModule(device, vert_module, nullptr);

    return {pipeline, layout};
}

auto create_predepth_pipeline(VkDevice device, PipelineCache &cache, VkDescriptorSetLayout bindless_layout,
                              const std::vector<uint32_t> &vert_code, const std::vector<uint32_t> &frag_code, VkFormat depth_format,
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
    vert_stage_ci.pName = "main_vs_mdi";

        auto frag_stage_ci = create_info<VkPipelineShaderStageCreateInfo>();
        frag_stage_ci.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        frag_stage_ci.module = frag_module;
        frag_stage_ci.pName = "fs_main";
        std::array stages = {vert_stage_ci ,frag_stage_ci,};

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

    std::array<VkVertexInputBindingDescription, 1> binding_descriptions{
            VkVertexInputBindingDescription{
                    .binding = 0,
                    .stride = sizeof(glm::vec3) + sizeof(u32),
                    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            }
    };

    std::array<VkVertexInputAttributeDescription, 2> attribute_descriptions{
            VkVertexInputAttributeDescription{
                    .location = 0,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = 0,
            },
            VkVertexInputAttributeDescription{
                    .location = 1,
                    .binding = 0,
                    .format = VK_FORMAT_R32_UINT,
                    .offset = sizeof(glm::vec3),
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
    ci.stageCount = static_cast<uint32_t>(stages.size());
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
    vkCreateGraphicsPipelines(device, cache, 1, &ci, nullptr, &pipeline);
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, "predepth_alpha_tested");

    // Cleanup local modules
    vkDestroyShaderModule(device, vert_module, nullptr);
    vkDestroyShaderModule(device, frag_module, nullptr);


    return {pipeline, layout};
}

auto create_tonemap_pipeline(VkDevice device, PipelineCache &cache, VkDescriptorSetLayout layout,
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
    vk_check(vkCreateGraphicsPipelines(device, cache, 1, &pipeline_info, nullptr, &pipeline));
        set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, "tonemap");

    vkDestroyShaderModule(device, vert_shader, nullptr);
    vkDestroyShaderModule(device, frag_shader, nullptr);

    return CompiledPipeline{pipeline, pipeline_layout};
}

auto create_gbuffer_pipeline(VkDevice device, PipelineCache &cache, VkDescriptorSetLayout bindless_layout,
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
                    .offset = offsetof(Vertex, normal),
            },
            VkVertexInputAttributeDescription{
                    .location = 2,
                    .binding = 0,
                    .format = VK_FORMAT_R32_UINT,
                    .offset = offsetof(Vertex, uvs),
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
   blend0.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                      VK_COLOR_COMPONENT_A_BIT;
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
    vk_check(vkCreateGraphicsPipelines(device, cache, 1, &gpci, nullptr, &pipeline));
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, "gbuffer_mrt");

    vkDestroyShaderModule(device, vert_module, nullptr);
    vkDestroyShaderModule(device, frag_module, nullptr);

    return CompiledPipeline{pipeline, pipeline_layout};
}

auto create_deferred_lighting_compute_pipeline(VkDevice device, PipelineCache &cache,
                                               VkDescriptorSetLayout bindless_layout, const std::vector<u32> &cs_code,
                                               std::string_view entry_name) -> CompiledPipeline {
    VkShaderModule cs_module{};
    {
        auto ci = create_info<VkShaderModuleCreateInfo>();
        ci.codeSize = cs_code.size() * sizeof(u32);
        ci.pCode = cs_code.data();
        vk_check(vkCreateShaderModule(device, &ci, nullptr, &cs_module));
    }

    VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(DeferredLightingPushConstants),
    };

    VkPipelineLayout layout{};
    {
        auto plci = create_info<VkPipelineLayoutCreateInfo>();
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &bindless_layout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &push_range;
        vk_check(vkCreatePipelineLayout(device, &plci, nullptr, &layout));
        set_debug_name(device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, layout, std::string(entry_name).c_str());
    }

    auto stage_ci = create_info<VkPipelineShaderStageCreateInfo>();
    stage_ci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_ci.module = cs_module;
    stage_ci.pName = entry_name.data();

    auto cpci = create_info<VkComputePipelineCreateInfo>();
    cpci.stage = stage_ci;
    cpci.layout = layout;

    VkPipeline pipeline{};
    vk_check(vkCreateComputePipelines(device, cache, 1, &cpci, nullptr, &pipeline));
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, std::string(entry_name).c_str());

    vkDestroyShaderModule(device, cs_module, nullptr);
    return CompiledPipeline{pipeline, layout};
}

auto create_deferred_lighting_graphics_pipeline(VkDevice device, PipelineCache &cache,
                                                VkDescriptorSetLayout bindless_layout,
                                                const std::vector<u32> &vert_code, const std::vector<u32> &frag_code,
                                                std::string_view vert_entry, std::string_view frag_entry,
                                                VkFormat color_format) -> CompiledPipeline {
    VkShaderModule vert_shader{};
    {
        auto ci = create_info<VkShaderModuleCreateInfo>();
        ci.codeSize = vert_code.size() * sizeof(u32);
        ci.pCode = vert_code.data();
        vk_check(vkCreateShaderModule(device, &ci, nullptr, &vert_shader));
        set_debug_name(device, VK_OBJECT_TYPE_SHADER_MODULE, vert_shader, "deferred_lighting_vs");
    }

    VkShaderModule frag_shader{};
    {
        auto ci = create_info<VkShaderModuleCreateInfo>();
        ci.codeSize = frag_code.size() * sizeof(u32);
        ci.pCode = frag_code.data();
        vk_check(vkCreateShaderModule(device, &ci, nullptr, &frag_shader));
        set_debug_name(device, VK_OBJECT_TYPE_SHADER_MODULE, frag_shader, "deferred_lighting_fs");
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
    vert_stage_ci.module = vert_shader;
    vert_stage_ci.pName = vert_entry.data();

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
    vk_check(vkCreateGraphicsPipelines(device, cache, 1, &gpci, nullptr, &pipeline));
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, "deferred_lighting_fs");

    vkDestroyShaderModule(device, vert_shader, nullptr);
    vkDestroyShaderModule(device, frag_shader, nullptr);

    return CompiledPipeline{pipeline, pipeline_layout};
}
