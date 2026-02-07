#include "ImGuiRenderer.hxx"
#include "BindlessHeadless.hxx"
#include "BindlessSet.hxx"
#include "Compiler.hxx"
#include "GlobalCommandContext.hxx"
#include "Pool.hxx"
#include "RenderContext.hxx"
#include "Types.hxx"

#include <imgui.h>
#include <implot.h>
#include <volk.h>


ImGuiRenderer::ImGuiRenderer(u32 image_count, RenderContext &c, GlobalCommandContext &cc, Compiler &comp,
                             FontChoice font) : ctx(c), command_ctx(cc), compiler(comp), drawables(image_count) {
    std::ignore = ImGui::CreateContext();
    std::ignore = ImPlot::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.BackendRendererName = "imgui-bindless-headless";
    io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;

    update_font(std::move(font));
}

ImGuiRenderer::~ImGuiRenderer() {
    ImGuiIO &io = ImGui::GetIO();
    io.Fonts->TexID = nullptr;
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
}

auto ImGuiRenderer::create_pipeline(ImGuiFramebuffer fb) -> tl::expected<CompiledPipeline, Error> {

    constexpr std::array<const std::string_view, 2> entry_points{"vs_main", "fs_main"};
    std::array<ReflectionData, entry_points.size()> reflection{};
    TRY_PROPAGATE(shaders, compiler.compile_from_file("shaders/gui.slang", entry_points, reflection),
                  "Could not compile gui shader");

    const std::array<u32, 1> data{0u};

    const VkSpecializationMapEntry is_color_space_nonlinear{
            .constantID = 0,
            .offset = 0,
            .size = sizeof(u32),
    };
    const std::array entries{is_color_space_nonlinear};
    VkSpecializationInfo spec_info{};
    spec_info.mapEntryCount = 1;
    spec_info.pMapEntries = entries.data();
    spec_info.dataSize = 1 * sizeof(u32);
    spec_info.pData = data.data();

    const std::array spec_infos{spec_info};


    VkShaderModule vert_shader{};
    {
        auto ci = create_info<VkShaderModuleCreateInfo>();
        ci.codeSize = shaders.at(0).size() * sizeof(u32);
        ci.pCode = shaders.at(0).data();
        vk_check(vkCreateShaderModule(ctx.get_device(), &ci, nullptr, &vert_shader));
    }

    VkShaderModule frag_shader{};
    {
        auto ci = create_info<VkShaderModuleCreateInfo>();
        ci.codeSize = shaders.at(1).size() * sizeof(u32);
        ci.pCode = shaders.at(1).data();
        vk_check(vkCreateShaderModule(ctx.get_device(), &ci, nullptr, &frag_shader));
    }

    VkPushConstantRange push_constant_range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(float) * 4 + sizeof(DeviceAddress) + sizeof(u32) * 2, // LRTB[4] + vb + textureId + samplerId
    };

    VkPipelineLayout pipeline_layout{};
    {
        auto plci = create_info<VkPipelineLayoutCreateInfo>();
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &ctx.bindless_set->layout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &push_constant_range;
        vk_check(vkCreatePipelineLayout(ctx.get_device(), &plci, nullptr, &pipeline_layout));
        set_debug_name(ctx.get_device(), VK_OBJECT_TYPE_PIPELINE_LAYOUT, pipeline_layout, "imgui");
    }

    auto vert_stage_ci = create_info<VkPipelineShaderStageCreateInfo>();
    vert_stage_ci.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_stage_ci.module = vert_shader;
    vert_stage_ci.pName = "vs_main";

    auto frag_stage_ci = create_info<VkPipelineShaderStageCreateInfo>();
    frag_stage_ci.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_stage_ci.module = frag_shader;
    frag_stage_ci.pName = "fs_main";
    frag_stage_ci.pSpecializationInfo = spec_infos.data();

    std::array shader_stages{vert_stage_ci, frag_stage_ci};
    auto vertex_input = create_info<VkPipelineVertexInputStateCreateInfo>();

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
    rasterization.lineWidth = 1.0f;

    auto multisample = create_info<VkPipelineMultisampleStateCreateInfo>();
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisample.sampleShadingEnable = VK_FALSE;
    multisample.minSampleShading = 1.0f;

    // ImGui uses alpha blending
    VkPipelineColorBlendAttachmentState color_blend_attachment{
            .blendEnable = VK_TRUE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                              VK_COLOR_COMPONENT_A_BIT,
    };

    auto color_blend = create_info<VkPipelineColorBlendStateCreateInfo>();
    color_blend.logicOpEnable = VK_FALSE;
    color_blend.attachmentCount = 1;
    color_blend.pAttachments = &color_blend_attachment;

    // ImGui doesn't use depth testing (as seen in end_frame)
    auto depth_stencil = create_info<VkPipelineDepthStencilStateCreateInfo>();
    depth_stencil.depthTestEnable = VK_FALSE;
    depth_stencil.depthWriteEnable = VK_FALSE;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;

    std::array dynamic_states{
            VK_DYNAMIC_STATE_VIEWPORT,     VK_DYNAMIC_STATE_SCISSOR,           VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
            VK_DYNAMIC_STATE_DEPTH_BOUNDS, VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
    };

    auto dynamic_state = create_info<VkPipelineDynamicStateCreateInfo>();
    dynamic_state.dynamicStateCount = static_cast<u32>(dynamic_states.size());
    dynamic_state.pDynamicStates = dynamic_states.data();

    // Dynamic rendering info
    VkFormat color_format = std::get<VkFormat>(fb);
    auto rendering_info = create_info<VkPipelineRenderingCreateInfo>();
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

    VkPipeline new_pipeline{};
    vk_check(vkCreateGraphicsPipelines(ctx.get_device(), *ctx.pipeline_cache, 1, &pipeline_info, nullptr,
                                       &new_pipeline));
    set_debug_name(ctx.get_device(), VK_OBJECT_TYPE_PIPELINE, new_pipeline, "imgui");

    vkDestroyShaderModule(ctx.get_device(), vert_shader, nullptr);
    vkDestroyShaderModule(ctx.get_device(), frag_shader, nullptr);

    return CompiledPipeline{
            .pipeline = new_pipeline,
            .layout = pipeline_layout,
    };
}


auto ImGuiRenderer::update_font(FontChoice f) -> void {
    ImGuiIO &io = ImGui::GetIO();
    ImFontConfig cfg = ImFontConfig();
    cfg.FontDataOwnedByAtlas = false;
    cfg.RasterizerMultiply = 1.5f;
    cfg.SizePixels = std::ceilf(f.size);
    cfg.PixelSnapH = true;
    cfg.OversampleH = 4;
    cfg.OversampleV = 4;
    ImFont *font = nullptr;
    if (std::filesystem::exists(f.font_path)) {
        auto path = f.font_path.string();
        font = io.Fonts->AddFontFromFileTTF(path.c_str(), cfg.SizePixels, &cfg);
    }
    io.Fonts->Flags |= ImFontAtlasFlags_NoPowerOfTwoHeight;
    unsigned char *pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    font_texture =
            Holder<TextureHandle>(ctx, ctx.create_texture(create_image_from_span_v2(
                                               ctx.allocator, command_ctx, width, height, VK_FORMAT_R8G8B8A8_UNORM,
                                               std::span(pixels, width * height * 4), "imgui_fonts")));
    io.Fonts->TexID = font_texture.index();
    io.FontDefault = font;
    auto ci = create_info<VkSamplerCreateInfo>();
    ci.magFilter = VK_FILTER_LINEAR;
    ci.minFilter = VK_FILTER_LINEAR;
    ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, // Change;
            ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, // Change;
            ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, // Change;
            ci.compareOp = VK_COMPARE_OP_ALWAYS;
    ci.maxLod = VK_LOD_CLAMP_NONE;
    ci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

    sampler = Holder{ctx, ctx.create_sampler(ci, "imgui_linear_clamp")};
}

auto ImGuiRenderer::begin_frame(ImGuiFramebuffer fb) -> void {

    const auto &dim = std::get<VkExtent2D>(fb);
    ImGuiIO &io = ImGui::GetIO();
    io.DisplaySize = ImVec2(dim.width / display_scale, dim.height / display_scale);
    io.DisplayFramebufferScale = ImVec2(display_scale, display_scale);
    io.IniFilename = nullptr;

    if (force_recompile) {
        auto created = create_pipeline(fb).value();
        pipeline = Holder{ctx, ctx.create_pipeline(std::move(created))};
        force_recompile = false;
    }

    if (pipeline.empty()) {
        auto created = create_pipeline(fb).value();
        pipeline = Holder{ctx, ctx.create_pipeline(std::move(created))};
    }

    ImGui::NewFrame();
}

auto ImGuiRenderer::end_frame(VkCommandBuffer cmd) -> void {
    ImGui::EndFrame();
    ImGui::Render();

    ImDrawData *dd = ImGui::GetDrawData();
    if (dd->TotalIdxCount == 0)
        return;

    const float fb_width = dd->DisplaySize.x * dd->FramebufferScale.x;
    const float fb_height = dd->DisplaySize.y * dd->FramebufferScale.y;

    auto &&[vp, sc] = viewport_scissors({
            static_cast<u32>(fb_width),
            static_cast<u32>(fb_height),
    });

    vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_ALWAYS);
    vkCmdSetDepthBounds(cmd, 0.0F, 1.0F);
    vkCmdSetDepthTestEnable(cmd, VK_FALSE);
    vkCmdSetDepthWriteEnable(cmd, VK_FALSE);

    vkCmdSetViewport(cmd, 0, 1, &vp);

    const float L = dd->DisplayPos.x;
    const float R = dd->DisplayPos.x + dd->DisplaySize.x;
    const float T = dd->DisplayPos.y;
    const float B = dd->DisplayPos.y + dd->DisplaySize.y;
    const ImVec2 clip_offset = dd->DisplayPos;
    const ImVec2 clip_scale = dd->FramebufferScale;

    auto &drawable = drawables.at(frame_index);
    frame_index = (frame_index + 1) % drawables.size();

    if (static_cast<i32>(drawable.index_count) < dd->TotalIdxCount) {
        auto buffer = Buffer::zeroes(ctx.allocator, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                     dd->TotalIdxCount * sizeof(ImDrawIdx), "imgui_index");
        drawable.index = Holder{ctx, ctx.create_buffer(std::move(buffer.value()))};
        drawable.index_count = static_cast<u32>(dd->TotalIdxCount);
    }

    if (static_cast<i32>(drawable.vertex_count) < dd->TotalVtxCount) {
        auto buffer = Buffer::zeroes(ctx.allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     dd->TotalVtxCount * sizeof(ImDrawVert), "imgui_vertex");
        drawable.vertex = Holder{ctx, ctx.create_buffer(std::move(buffer.value()))};
        drawable.vertex_count = static_cast<u32>(dd->TotalVtxCount);
    }

    {
        static std::vector<ImDrawVert> all_vtx;
        static std::vector<ImDrawIdx> all_itx;

        all_vtx.clear();
        all_itx.clear();

        if (all_vtx.capacity() < static_cast<size_t>(dd->TotalVtxCount)) {
            all_vtx.reserve(dd->TotalVtxCount);
        }
        if (all_itx.capacity() < static_cast<size_t>(dd->TotalIdxCount)) {
            all_itx.reserve(dd->TotalIdxCount);
        }

        for (int n = 0; n < dd->CmdListsCount; n++) {
            const auto *imgui_cmd = dd->CmdLists[n];
            all_vtx.insert(all_vtx.end(), imgui_cmd->VtxBuffer.Data,
                           imgui_cmd->VtxBuffer.Data + imgui_cmd->VtxBuffer.Size);
            all_itx.insert(all_itx.end(), imgui_cmd->IdxBuffer.Data,
                           imgui_cmd->IdxBuffer.Data + imgui_cmd->IdxBuffer.Size);
        }

        ctx.buffers.get(drawable.vertex)->write_slice(ctx.allocator, std::span{all_vtx}, 0);
        ctx.buffers.get(drawable.index)->write_slice(ctx.allocator, std::span{all_itx}, 0);
    }
    auto &&[itx, vtx] = ctx.buffers.get_multiple(drawable.index, drawable.vertex);
    auto *pipe = ctx.pipeline_pool.get(pipeline);

    vkCmdBindIndexBuffer(cmd, itx->buffer(), 0, VK_INDEX_TYPE_UINT16);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe->layout, 0, 1, &ctx.bindless_set->set, 0,
                            nullptr);

    u32 index_offset = 0;
    u32 vertex_offset = 0;
    for (int n = 0; n < dd->CmdListsCount; n++) {
        const auto *command_list = dd->CmdLists[n];
        for (int cmd_i = 0; cmd_i < command_list->CmdBuffer.Size; cmd_i++) {
            const auto &imgui_cmd = command_list->CmdBuffer[cmd_i];
            ImVec2 clipMin((imgui_cmd.ClipRect.x - clip_offset.x) * clip_scale.x,
                           (imgui_cmd.ClipRect.y - clip_offset.y) * clip_scale.y);
            ImVec2 clipMax((imgui_cmd.ClipRect.z - clip_offset.x) * clip_scale.x,
                           (imgui_cmd.ClipRect.w - clip_offset.y) * clip_scale.y);
            if (clipMin.x < 0.0f)
                clipMin.x = 0.0f;
            if (clipMin.y < 0.0f)
                clipMin.y = 0.0f;
            if (clipMax.x > fb_width)
                clipMax.x = fb_width;
            if (clipMax.y > fb_height)
                clipMax.y = fb_height;
            if (clipMax.x <= clipMin.x || clipMax.y <= clipMin.y)
                continue;
            struct VulkanImguiBindData {
                std::array<float, 4> LRTB{};
                const DeviceAddress vb;
                u32 texture_id = 0;
                u32 sampler_id = 0;
            } bindData = {
                    .LRTB = {L, R, T, B},
                    .vb = ctx.device_address(drawable.vertex),
                    .texture_id = static_cast<u32>(imgui_cmd.GetTexID()),
                    .sampler_id = sampler.index(),
            };
            vkCmdPushConstants(cmd, pipe->layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(VulkanImguiBindData), &bindData);
            VkRect2D scissor{
                    .offset = {static_cast<i32>(clipMin.x), static_cast<i32>(clipMin.y)},
                    .extent = {static_cast<u32>(clipMax.x - clipMin.x), static_cast<u32>(clipMax.y - clipMin.y)},
            };
            vkCmdSetScissor(cmd, 0, 1, &scissor);
            vkCmdDrawIndexed(cmd, imgui_cmd.ElemCount, 1, index_offset + imgui_cmd.IdxOffset,
                             vertex_offset + imgui_cmd.VtxOffset, 0);
        }
        index_offset += command_list->IdxBuffer.Size;
        vertex_offset += command_list->VtxBuffer.Size;
    }
}

auto ImGui_KeyToImGuiKey(int key) -> ImGuiKey {
    switch (key) {
        case GLFW_KEY_TAB:
            return ImGuiKey_Tab;
        case GLFW_KEY_LEFT:
            return ImGuiKey_LeftArrow;
        case GLFW_KEY_RIGHT:
            return ImGuiKey_RightArrow;
        case GLFW_KEY_UP:
            return ImGuiKey_UpArrow;
        case GLFW_KEY_DOWN:
            return ImGuiKey_DownArrow;
        case GLFW_KEY_PAGE_UP:
            return ImGuiKey_PageUp;
        case GLFW_KEY_PAGE_DOWN:
            return ImGuiKey_PageDown;
        case GLFW_KEY_HOME:
            return ImGuiKey_Home;
        case GLFW_KEY_END:
            return ImGuiKey_End;
        case GLFW_KEY_INSERT:
            return ImGuiKey_Insert;
        case GLFW_KEY_DELETE:
            return ImGuiKey_Delete;
        case GLFW_KEY_BACKSPACE:
            return ImGuiKey_Backspace;
        case GLFW_KEY_SPACE:
            return ImGuiKey_Space;
        case GLFW_KEY_ENTER:
            return ImGuiKey_Enter;
        case GLFW_KEY_ESCAPE:
            return ImGuiKey_Escape;
        case GLFW_KEY_APOSTROPHE:
            return ImGuiKey_Apostrophe;
        case GLFW_KEY_COMMA:
            return ImGuiKey_Comma;
        case GLFW_KEY_MINUS:
            return ImGuiKey_Minus;
        case GLFW_KEY_PERIOD:
            return ImGuiKey_Period;
        case GLFW_KEY_SLASH:
            return ImGuiKey_Slash;
        case GLFW_KEY_SEMICOLON:
            return ImGuiKey_Semicolon;
        case GLFW_KEY_EQUAL:
            return ImGuiKey_Equal;
        case GLFW_KEY_LEFT_BRACKET:
            return ImGuiKey_LeftBracket;
        case GLFW_KEY_BACKSLASH:
            return ImGuiKey_Backslash;
        case GLFW_KEY_RIGHT_BRACKET:
            return ImGuiKey_RightBracket;
        case GLFW_KEY_GRAVE_ACCENT:
            return ImGuiKey_GraveAccent;
        case GLFW_KEY_CAPS_LOCK:
            return ImGuiKey_CapsLock;
        case GLFW_KEY_SCROLL_LOCK:
            return ImGuiKey_ScrollLock;
        case GLFW_KEY_NUM_LOCK:
            return ImGuiKey_NumLock;
        case GLFW_KEY_PRINT_SCREEN:
            return ImGuiKey_PrintScreen;
        case GLFW_KEY_PAUSE:
            return ImGuiKey_Pause;
        case GLFW_KEY_KP_0:
            return ImGuiKey_Keypad0;
        case GLFW_KEY_KP_1:
            return ImGuiKey_Keypad1;
        case GLFW_KEY_KP_2:
            return ImGuiKey_Keypad2;
        case GLFW_KEY_KP_3:
            return ImGuiKey_Keypad3;
        case GLFW_KEY_KP_4:
            return ImGuiKey_Keypad4;
        case GLFW_KEY_KP_5:
            return ImGuiKey_Keypad5;
        case GLFW_KEY_KP_6:
            return ImGuiKey_Keypad6;
        case GLFW_KEY_KP_7:
            return ImGuiKey_Keypad7;
        case GLFW_KEY_KP_8:
            return ImGuiKey_Keypad8;
        case GLFW_KEY_KP_9:
            return ImGuiKey_Keypad9;
        case GLFW_KEY_KP_DECIMAL:
            return ImGuiKey_KeypadDecimal;
        case GLFW_KEY_KP_DIVIDE:
            return ImGuiKey_KeypadDivide;
        case GLFW_KEY_KP_MULTIPLY:
            return ImGuiKey_KeypadMultiply;
        case GLFW_KEY_KP_SUBTRACT:
            return ImGuiKey_KeypadSubtract;
        case GLFW_KEY_KP_ADD:
            return ImGuiKey_KeypadAdd;
        case GLFW_KEY_KP_ENTER:
            return ImGuiKey_KeypadEnter;
        case GLFW_KEY_KP_EQUAL:
            return ImGuiKey_KeypadEqual;
        case GLFW_KEY_LEFT_SHIFT:
            return ImGuiKey_LeftShift;
        case GLFW_KEY_LEFT_CONTROL:
            return ImGuiKey_LeftCtrl;
        case GLFW_KEY_LEFT_ALT:
            return ImGuiKey_LeftAlt;
        case GLFW_KEY_LEFT_SUPER:
            return ImGuiKey_LeftSuper;
        case GLFW_KEY_RIGHT_SHIFT:
            return ImGuiKey_RightShift;
        case GLFW_KEY_RIGHT_CONTROL:
            return ImGuiKey_RightCtrl;
        case GLFW_KEY_RIGHT_ALT:
            return ImGuiKey_RightAlt;
        case GLFW_KEY_RIGHT_SUPER:
            return ImGuiKey_RightSuper;
        case GLFW_KEY_MENU:
            return ImGuiKey_Menu;
        case GLFW_KEY_0:
            return ImGuiKey_0;
        case GLFW_KEY_1:
            return ImGuiKey_1;
        case GLFW_KEY_2:
            return ImGuiKey_2;
        case GLFW_KEY_3:
            return ImGuiKey_3;
        case GLFW_KEY_4:
            return ImGuiKey_4;
        case GLFW_KEY_5:
            return ImGuiKey_5;
        case GLFW_KEY_6:
            return ImGuiKey_6;
        case GLFW_KEY_7:
            return ImGuiKey_7;
        case GLFW_KEY_8:
            return ImGuiKey_8;
        case GLFW_KEY_9:
            return ImGuiKey_9;
        case GLFW_KEY_A:
            return ImGuiKey_A;
        case GLFW_KEY_B:
            return ImGuiKey_B;
        case GLFW_KEY_C:
            return ImGuiKey_C;
        case GLFW_KEY_D:
            return ImGuiKey_D;
        case GLFW_KEY_E:
            return ImGuiKey_E;
        case GLFW_KEY_F:
            return ImGuiKey_F;
        case GLFW_KEY_G:
            return ImGuiKey_G;
        case GLFW_KEY_H:
            return ImGuiKey_H;
        case GLFW_KEY_I:
            return ImGuiKey_I;
        case GLFW_KEY_J:
            return ImGuiKey_J;
        case GLFW_KEY_K:
            return ImGuiKey_K;
        case GLFW_KEY_L:
            return ImGuiKey_L;
        case GLFW_KEY_M:
            return ImGuiKey_M;
        case GLFW_KEY_N:
            return ImGuiKey_N;
        case GLFW_KEY_O:
            return ImGuiKey_O;
        case GLFW_KEY_P:
            return ImGuiKey_P;
        case GLFW_KEY_Q:
            return ImGuiKey_Q;
        case GLFW_KEY_R:
            return ImGuiKey_R;
        case GLFW_KEY_S:
            return ImGuiKey_S;
        case GLFW_KEY_T:
            return ImGuiKey_T;
        case GLFW_KEY_U:
            return ImGuiKey_U;
        case GLFW_KEY_V:
            return ImGuiKey_V;
        case GLFW_KEY_W:
            return ImGuiKey_W;
        case GLFW_KEY_X:
            return ImGuiKey_X;
        case GLFW_KEY_Y:
            return ImGuiKey_Y;
        case GLFW_KEY_Z:
            return ImGuiKey_Z;
        case GLFW_KEY_F1:
            return ImGuiKey_F1;
        case GLFW_KEY_F2:
            return ImGuiKey_F2;
        case GLFW_KEY_F3:
            return ImGuiKey_F3;
        case GLFW_KEY_F4:
            return ImGuiKey_F4;
        case GLFW_KEY_F5:
            return ImGuiKey_F5;
        case GLFW_KEY_F6:
            return ImGuiKey_F6;
        case GLFW_KEY_F7:
            return ImGuiKey_F7;
        case GLFW_KEY_F8:
            return ImGuiKey_F8;
        case GLFW_KEY_F9:
            return ImGuiKey_F9;
        case GLFW_KEY_F10:
            return ImGuiKey_F10;
        case GLFW_KEY_F11:
            return ImGuiKey_F11;
        case GLFW_KEY_F12:
            return ImGuiKey_F12;
        default:
            return ImGuiKey_None;
    }
}
