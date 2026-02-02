#include "Pipelines.hxx"
#include "BindlessHeadless.hxx"
#include "PipelineCache.hxx"
#include "Mesh.hxx"

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


    const VkComputePipelineCreateInfo cpci{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
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
    VkShaderModuleCreateInfo create_info = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                            .pNext = nullptr,
                                            .flags = 0,
                                            .codeSize = vert_code.size() * sizeof(u32),
                                            .pCode = vert_code.data()};
    vk_check(vkCreateShaderModule(device, &create_info, nullptr, &vert_module));

    std::array stages = {
            VkPipelineShaderStageCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .stage = VK_SHADER_STAGE_VERTEX_BIT,
                    .module = vert_module,
                    .pName = "main_vs_mdi",
                    .pSpecializationInfo = nullptr,
            },
    };

    // 2. Pipeline Layout (Inherit bindless + push constants)
    VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .offset = 0,
            .size = sizeof(PredepthPushConstants),
    };

    VkPipelineLayoutCreateInfo layout_ci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &bindless_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
    };
    VkPipelineLayout layout;
    vkCreatePipelineLayout(device, &layout_ci, nullptr, &layout);
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, layout, "predepth");


    // 3. Specialized Depth State
    VkPipelineDepthStencilStateCreateInfo ds{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL, // Reverse-Z: Near is 1.0, Far is 0.0
            .depthBoundsTestEnable = VkBool32{},
            .stencilTestEnable = VkBool32{},
            .front = {},
            .back = {},
            .minDepthBounds = 0.0f,
            .maxDepthBounds = 1.0f,
    };

    // 4. No Color Attachments (The secret to Pre-Depth speed)
    VkPipelineColorBlendStateCreateInfo cb{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .logicOpEnable = VkBool32{},
            .logicOp = VK_LOGIC_OP_MAX_ENUM,
            .attachmentCount = 0,
            .pAttachments = nullptr,
            .blendConstants = {0, 0, 0, 0},
    };

    // 5. Rasterization (Ensure Back-Face Culling is ON)
    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    // 6. Dynamic Rendering Info
    VkPipelineRenderingCreateInfo rendering_info{};
    rendering_info.depthAttachmentFormat = depth_format;
    rendering_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;

    // Viewport/Scissor setup (Standard)
    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = samples;
    std::array dynamic_states = {VK_DYNAMIC_STATE_VIEWPORT,         VK_DYNAMIC_STATE_SCISSOR,
                                 VK_DYNAMIC_STATE_DEPTH_COMPARE_OP, VK_DYNAMIC_STATE_DEPTH_BOUNDS,
                                 VK_DYNAMIC_STATE_CULL_MODE,        VK_DYNAMIC_STATE_FRONT_FACE};
    VkPipelineDynamicStateCreateInfo dy{};
    dy.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dy.dynamicStateCount = static_cast<u32>(dynamic_states.size());
    dy.pDynamicStates = dynamic_states.data();

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

    VkPipelineInputAssemblyStateCreateInfo assembly_state{};
    assembly_state.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    assembly_state.primitiveRestartEnable = VK_FALSE;

    VkGraphicsPipelineCreateInfo ci{.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                                    .pNext = &rendering_info,
                                    .flags = 0,
                                    .stageCount = static_cast<uint32_t>(stages.size()),
                                    .pStages = stages.data(),
                                    .pVertexInputState = &vertex_input,
                                    .pInputAssemblyState = &assembly_state,
                                    .pTessellationState = nullptr,
                                    .pViewportState = &vp,
                                    .pRasterizationState = &rs,
                                    .pMultisampleState = &ms,
                                    .pDepthStencilState = &ds,
                                    .pColorBlendState = &cb,
                                    .pDynamicState = &dy,
                                    .layout = layout,
                                    .renderPass = VK_NULL_HANDLE,
                                    .subpass = 0,
                                    .basePipelineHandle = VK_NULL_HANDLE,
                                    .basePipelineIndex = -1};

    VkPipeline pipeline;
    vkCreateGraphicsPipelines(device, cache, 1, &ci, nullptr, &pipeline);
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, "predepth");


    // Cleanup local modules
    vkDestroyShaderModule(device, vert_module, nullptr);

    return {pipeline, layout};
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

auto create_gbuffer_pipeline(VkDevice device,
                             PipelineCache& cache,
                             VkDescriptorSetLayout bindless_layout,
                             const std::vector<u32>& vert_code,
                             const std::vector<u32>& frag_code,
                             VkFormat gbuffer0_format,
                             VkFormat gbuffer1_format,
                             VkFormat gbuffer2_format,
                             VkFormat depth_format) -> CompiledPipeline
{
    VkShaderModule vert_module{};
    {
        VkShaderModuleCreateInfo ci{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = vert_code.size() * sizeof(u32),
            .pCode = vert_code.data(),
        };
        vk_check(vkCreateShaderModule(device, &ci, nullptr, &vert_module));
    }

    VkShaderModule frag_module{};
    {
        VkShaderModuleCreateInfo ci{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = frag_code.size() * sizeof(u32),
            .pCode = frag_code.data(),
        };
        vk_check(vkCreateShaderModule(device, &ci, nullptr, &frag_module));
    }

    std::array stages{
        VkPipelineShaderStageCreateInfo{
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vert_module,
            .pName  = "main_vs_mdi",
        },
        VkPipelineShaderStageCreateInfo{
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = frag_module,
            .pName  = "main_fs_mdi",
        },
    };

    // Push constants for the GBuffer pass (your Slang GBufferPC)
    VkPushConstantRange push_range{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = sizeof(RenderingPushConstants),
    };

    VkPipelineLayout pipeline_layout{};
    {
        VkPipelineLayoutCreateInfo plci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &bindless_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
        };
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
            .binding  = 0,
            .format   = VK_FORMAT_R32G32B32_SFLOAT,
            .offset   = offsetof(Vertex, position),
        },
        VkVertexInputAttributeDescription{
            .location = 1,
            .binding  = 0,
            .format   = VK_FORMAT_R32_UINT,
            .offset   = offsetof(Vertex, normal),
        },
        VkVertexInputAttributeDescription{
            .location = 2,
            .binding  = 0,
            .format   = VK_FORMAT_R32_UINT,
            .offset   = offsetof(Vertex, uvs),
        },
        VkVertexInputAttributeDescription{
            .location = 3,
            .binding  = 0,
            .format   = VK_FORMAT_R32_UINT,
            .offset   = offsetof(Vertex, tangent),
        },
        VkVertexInputAttributeDescription{
            .location = 4,
            .binding  = 0,
            .format   = VK_FORMAT_R32_UINT,
            .offset   = offsetof(Vertex, bitangent),
        },
    };

    VkPipelineVertexInputStateCreateInfo vi{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = static_cast<u32>(bindings.size()),
        .pVertexBindingDescriptions = bindings.data(),
        .vertexAttributeDescriptionCount = static_cast<u32>(attrs.size()),
        .pVertexAttributeDescriptions = attrs.data(),
    };

    VkPipelineInputAssemblyStateCreateInfo ia{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    VkPipelineViewportStateCreateInfo vp{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount  = 1,
    };

    VkPipelineRasterizationStateCreateInfo rs{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };

    VkPipelineMultisampleStateCreateInfo ms{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    // You’re doing predepth, so keep depth test ON, depth write OFF.
    // Compare op is dynamic in your engine, but set a sane default.
    VkPipelineDepthStencilStateCreateInfo ds{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_FALSE,
        .depthCompareOp = VK_COMPARE_OP_EQUAL, // matches your vkCmdSetDepthCompareOp(EQUAL)
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
        .minDepthBounds = 0.0f,
        .maxDepthBounds = 1.0f,
    };

    // MRT: 3 attachments, no blending for gbuffer writes.
    VkPipelineColorBlendAttachmentState blend0{
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    std::array<VkPipelineColorBlendAttachmentState, 3> blends{blend0, blend0, blend0};

    VkPipelineColorBlendStateCreateInfo cb{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = static_cast<u32>(blends.size()),
        .pAttachments = blends.data(),
    };

    std::array<VkDynamicState, 6> dyn_states{
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
        VK_DYNAMIC_STATE_DEPTH_BOUNDS,
        VK_DYNAMIC_STATE_CULL_MODE,
        VK_DYNAMIC_STATE_FRONT_FACE,
    };

    VkPipelineDynamicStateCreateInfo dy{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<u32>(dyn_states.size()),
        .pDynamicStates = dyn_states.data(),
    };

    std::array<VkFormat, 3> color_formats{gbuffer0_format, gbuffer1_format, gbuffer2_format};

    VkPipelineRenderingCreateInfo rendering_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = static_cast<u32>(color_formats.size()),
        .pColorAttachmentFormats = color_formats.data(),
        .depthAttachmentFormat = depth_format,
        .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
    };

    VkGraphicsPipelineCreateInfo gpci{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &rendering_info,
        .stageCount = static_cast<u32>(stages.size()),
        .pStages = stages.data(),
        .pVertexInputState = &vi,
        .pInputAssemblyState = &ia,
        .pViewportState = &vp,
        .pRasterizationState = &rs,
        .pMultisampleState = &ms,
        .pDepthStencilState = &ds,
        .pColorBlendState = &cb,
        .pDynamicState = &dy,
        .layout = pipeline_layout,
        .renderPass = VK_NULL_HANDLE,
        .subpass = 0,
    };

    VkPipeline pipeline{};
    vk_check(vkCreateGraphicsPipelines(device, cache, 1, &gpci, nullptr, &pipeline));
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, "gbuffer_mrt");

    vkDestroyShaderModule(device, vert_module, nullptr);
    vkDestroyShaderModule(device, frag_module, nullptr);

    return CompiledPipeline{pipeline, pipeline_layout};
}

auto create_deferred_lighting_compute_pipeline(VkDevice device,
                                               PipelineCache& cache,
                                               VkDescriptorSetLayout bindless_layout,
                                               const std::vector<u32>& cs_code,
                                               std::string_view entry_name) -> CompiledPipeline
{
    VkShaderModule cs_module{};
    {
        VkShaderModuleCreateInfo ci{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = cs_code.size() * sizeof(u32),
            .pCode = cs_code.data(),
        };
        vk_check(vkCreateShaderModule(device, &ci, nullptr, &cs_module));
    }

    VkPushConstantRange push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(DeferredLightingPushConstants),
    };

    VkPipelineLayout layout{};
    {
        VkPipelineLayoutCreateInfo plci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &bindless_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
        };
        vk_check(vkCreatePipelineLayout(device, &plci, nullptr, &layout));
        set_debug_name(device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, layout, std::string(entry_name).c_str());
    }

    VkComputePipelineCreateInfo cpci{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = cs_module,
            .pName = entry_name.data(),
            .pSpecializationInfo = nullptr,
        },
        .layout = layout,
    };

    VkPipeline pipeline{};
    vk_check(vkCreateComputePipelines(device, cache, 1, &cpci, nullptr, &pipeline));
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, std::string(entry_name).c_str());

    vkDestroyShaderModule(device, cs_module, nullptr);
    return CompiledPipeline{pipeline, layout};
}

auto create_deferred_lighting_graphics_pipeline(
        VkDevice device,
        PipelineCache &cache,
        VkDescriptorSetLayout bindless_layout,
        const std::vector<u32> &vert_code,
        const std::vector<u32> &frag_code,
        std::string_view vert_entry,
        std::string_view frag_entry,
        VkFormat color_format) -> CompiledPipeline
{
    VkShaderModule vert_shader{};
    {
        VkShaderModuleCreateInfo ci{
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .codeSize = vert_code.size() * sizeof(u32),
                .pCode = vert_code.data(),
        };
        vk_check(vkCreateShaderModule(device, &ci, nullptr, &vert_shader));
        set_debug_name(device, VK_OBJECT_TYPE_SHADER_MODULE, vert_shader, "deferred_lighting_vs");
    }

    VkShaderModule frag_shader{};
    {
        VkShaderModuleCreateInfo ci{
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .codeSize = frag_code.size() * sizeof(u32),
                .pCode = frag_code.data(),
        };
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
        VkPipelineLayoutCreateInfo plci{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .setLayoutCount = 1,
                .pSetLayouts = &bindless_layout,
                .pushConstantRangeCount = 1,
                .pPushConstantRanges = &push_constant_range,
        };
        vk_check(vkCreatePipelineLayout(device, &plci, nullptr, &pipeline_layout));
        set_debug_name(device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, pipeline_layout, "deferred_lighting_layout");
    }

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
            },
    };

    // Fullscreen triangle: no vertex buffers.
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

    // No culling for fullscreen triangle.
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

    // Deferred lighting pass: no depth test/write.
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
            .minDepthBounds = 0.0f,
            .maxDepthBounds = 1.0f,
    };

    VkPipelineColorBlendAttachmentState color_blend_attachment{
            .blendEnable = VK_FALSE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask =
                    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
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
            .blendConstants = {0, 0, 0, 0},
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

    VkGraphicsPipelineCreateInfo gpci{
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
    vk_check(vkCreateGraphicsPipelines(device, cache, 1, &gpci, nullptr, &pipeline));
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE, pipeline, "deferred_lighting_fs");

    vkDestroyShaderModule(device, vert_shader, nullptr);
    vkDestroyShaderModule(device, frag_shader, nullptr);

    return CompiledPipeline{pipeline, pipeline_layout};
}
