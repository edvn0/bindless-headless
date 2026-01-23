#include "Pipelines.hxx"
#include "BindlessHeadless.hxx"
#include "PipelineCache.hxx"

#include <glm/glm.hpp>

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
            .size = sizeof(PointLightCullingPushConstants),
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

    VkSpecializationInfo waves_per_group_spec_info{};
    VkSpecializationMapEntry waves_per_group_spec_map_entry{
            .constantID = 0,
            .offset = 0,
            .size = sizeof(u32),
    };
    waves_per_group_spec_info.mapEntryCount = 1;
    waves_per_group_spec_info.pMapEntries = &waves_per_group_spec_map_entry;
    waves_per_group_spec_info.dataSize = sizeof(u32);
    waves_per_group_spec_info.pData = &MAX_WAVES_PER_GROUP;

    VkSpecializationInfo threads_per_group_spec_info{};
    VkSpecializationMapEntry threads_per_group_spec_map_entry{
            .constantID = 1,
            .offset = 0,
            .size = sizeof(u32),
    };
    threads_per_group_spec_info.mapEntryCount = 1;
    threads_per_group_spec_info.pMapEntries = &threads_per_group_spec_map_entry;
    threads_per_group_spec_info.dataSize = sizeof(u32);
    threads_per_group_spec_info.pData = &THREADS_PER_GROUP;

    std::array spec_infos{waves_per_group_spec_info, threads_per_group_spec_info};


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
                                                     .pSpecializationInfo = spec_infos.data(),
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

auto create_predepth_pipeline(VkDevice device, PipelineCache &cache, VkDescriptorSetLayout bindless_layout,
                              const std::vector<uint32_t> &vert_code, const std::vector<uint32_t> &frag_code,
                              VkFormat depth_format) -> CompiledPipeline {
    VkShaderModule vert_module{};
    VkShaderModuleCreateInfo create_info = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                            .pNext = nullptr,
                                            .flags = 0,
                                            .codeSize = vert_code.size() * sizeof(u32),
                                            .pCode = vert_code.data()};
    vk_check(vkCreateShaderModule(device, &create_info, nullptr, &vert_module));

    VkShaderModule frag_module{};
    create_info = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                   .pNext = nullptr,
                   .flags = 0,
                   .codeSize = frag_code.size() * sizeof(u32),
                   .pCode = frag_code.data()};
    vk_check(vkCreateShaderModule(device, &create_info, nullptr, &frag_module));

    std::array stages = {
            VkPipelineShaderStageCreateInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                            .pNext = nullptr,
                                            .flags = 0,
                                            .stage = VK_SHADER_STAGE_VERTEX_BIT,
                                            .module = vert_module,
                                            .pName = "main_vs",
                                            .pSpecializationInfo = nullptr},
            VkPipelineShaderStageCreateInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                            .pNext = nullptr,
                                            .flags = 0,
                                            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                                            .module = frag_module,
                                            .pName = "main_fs",
                                            .pSpecializationInfo = nullptr},
    };

    // 2. Pipeline Layout (Inherit bindless + push constants)
    VkPushConstantRange push_range{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
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

    std::array<VkVertexInputBindingDescription, 1> binding_descriptions{
            VkVertexInputBindingDescription{
                    .binding = 0,
                    .stride = sizeof(glm::vec3),
                    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            },
    };

    std::array<VkVertexInputAttributeDescription, 1> attribute_descriptions{
            VkVertexInputAttributeDescription{
                    .location = 0,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = 0,
            },
    };

    VkPipelineVertexInputStateCreateInfo vertex_input{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .vertexBindingDescriptionCount = static_cast<u32>(binding_descriptions.size()),
            .pVertexBindingDescriptions = binding_descriptions.data(),
            .vertexAttributeDescriptionCount = static_cast<u32>(attribute_descriptions.size()),
            .pVertexAttributeDescriptions = attribute_descriptions.data(),
    };

    VkPipelineInputAssemblyStateCreateInfo assembly_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE,
    };

    VkPipelineTessellationStateCreateInfo tesselation_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
            .patchControlPoints = 0,
    };

    VkGraphicsPipelineCreateInfo ci{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = &rendering_info,
            .stageCount = static_cast<uint32_t>(stages.size()),
            .pStages = stages.data(),
            .pVertexInputState = &vertex_input,
            .pInputAssemblyState = &assembly_state,
            .pTessellationState = &tesselation_state,
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
    vkDestroyShaderModule(device, vert_module, nullptr);
    vkDestroyShaderModule(device, frag_module, nullptr);

    return {pipeline, layout};
}

auto create_mesh_pipeline(VkDevice device, PipelineCache &cache, VkDescriptorSetLayout layout,
                          const std::vector<u32> &vert_code, const std::vector<u32> &frag_code, VkFormat color_format)
        -> CompiledPipeline {
    VkShaderModule vert_module{};
    VkShaderModuleCreateInfo task_create_info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                              .pNext = nullptr,
                                              .flags = 0,
                                              .codeSize = vert_code.size() * sizeof(u32),
                                              .pCode = vert_code.data()};
    vk_check(vkCreateShaderModule(device, &task_create_info, nullptr, &vert_module));

    VkShaderModule frag_module{};
    VkShaderModuleCreateInfo frag_create_info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                              .pNext = nullptr,
                                              .flags = 0,
                                              .codeSize = frag_code.size() * sizeof(u32),
                                              .pCode = frag_code.data()};
    vk_check(vkCreateShaderModule(device, &frag_create_info, nullptr, &frag_module));

    VkPushConstantRange push_constant_range{
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT,
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
             .stage = VK_SHADER_STAGE_VERTEX_BIT,
             .module = vert_module,
             .pName = "main_vs"},
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
            .cullMode = VK_CULL_MODE_BACK_BIT,
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
            .minDepthBounds = 0.0f,
            .maxDepthBounds = 1.0f,
    };

    VkPipelineColorBlendAttachmentState color_blend_attachment{
            .blendEnable = VK_TRUE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
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
    std::array<VkVertexInputBindingDescription, 1> binding_descriptions{
            VkVertexInputBindingDescription{
                    .binding = 0,
                    .stride = sizeof(glm::vec3) + sizeof(u32)*2,
                    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            },
    };

    std::array attribute_descriptions{
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
                    .offset = 0,
            },
            VkVertexInputAttributeDescription{
                    .location = 2,
                    .binding = 0,
                    .format = VK_FORMAT_R32_UINT,
                    .offset = 0,
            },
    };

    VkPipelineVertexInputStateCreateInfo vertex_input{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .vertexBindingDescriptionCount = static_cast<u32>(binding_descriptions.size()),
            .pVertexBindingDescriptions = binding_descriptions.data(),
            .vertexAttributeDescriptionCount = static_cast<u32>(attribute_descriptions.size()),
            .pVertexAttributeDescriptions = attribute_descriptions.data(),
    };

    VkPipelineInputAssemblyStateCreateInfo assembly_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE,
    };

    VkGraphicsPipelineCreateInfo pipeline_info{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = &rendering_info,
            .flags = 0,
            .stageCount = static_cast<u32>(stages.size()),
            .pStages = stages.data(),
            .pVertexInputState = &vertex_input,
            .pInputAssemblyState = &assembly_state,
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


    vkDestroyShaderModule(device, vert_module, nullptr);
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
