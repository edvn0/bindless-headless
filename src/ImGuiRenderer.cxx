#include "ImGuiRenderer.hxx"
#include "BindlessHeadless.hxx"
#include "BindlessSet.hxx"
#include "Compiler.hxx"
#include "GlobalCommandContext.hxx"
#include "Pool.hxx"
#include "RenderContext.hxx"
#include "Swapchain.hxx"
#include "Types.hxx"

#include <backends/imgui_impl_glfw.h>
#include <filesystem>
#include <imgui.h>
#include <implot.h>
#include <bit>
#include <volk.h>

#include <unordered_map>

namespace {

    struct ImGuiViewportRenderTarget {
        GLFWwindow *window{nullptr};
        VkSurfaceKHR surface{VK_NULL_HANDLE};
        Swapchain swapchain{};

        VkCommandPool command_pool{VK_NULL_HANDLE};
        std::vector<VkCommandBuffer> command_buffers{};
        std::vector<VkFence> fences{};

        // local frame index for this viewport's command buffers/fences
        u32 frame_index{0};
    };

    static std::unordered_map<GLFWwindow *, ImGuiViewportRenderTarget> viewport_targets{};

    static auto destroy_viewport_target(RenderContext &ctx, ImGuiViewportRenderTarget &rt) -> void {
        VkDevice device = ctx.get_device();

        if (rt.command_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, rt.command_pool, nullptr);
            rt.command_pool = VK_NULL_HANDLE;
        }

        for (auto f: rt.fences) {
            if (f != VK_NULL_HANDLE) {
                vkDestroyFence(device, f, nullptr);
            }
        }
        rt.fences.clear();
        rt.command_buffers.clear();

        rt.swapchain.destroy();

        if (rt.surface != VK_NULL_HANDLE) {
            vkDestroySurfaceKHR(ctx.get_instance(), rt.surface, nullptr);
            rt.surface = VK_NULL_HANDLE;
        }

        rt.window = nullptr;
        rt.frame_index = 0;
    }

    static auto ensure_viewport_command_resources(RenderContext &ctx, ImGuiViewportRenderTarget &rt, u32 cmd_count)
            -> void {
        VkDevice device = ctx.get_device();

        if (rt.command_pool == VK_NULL_HANDLE) {
            VkCommandPoolCreateInfo cpci{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                    .queueFamilyIndex = ctx.queues.graphics.family_index,
            };
            vk_check(vkCreateCommandPool(device, &cpci, nullptr, &rt.command_pool));
            set_debug_name(device, VK_OBJECT_TYPE_COMMAND_POOL, rt.command_pool, "imgui_viewport_cmd_pool");
        }

        if (rt.command_buffers.size() != cmd_count) {
            rt.command_buffers.assign(cmd_count, VK_NULL_HANDLE);

            VkCommandBufferAllocateInfo cbai{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                    .pNext = nullptr,
                    .commandPool = rt.command_pool,
                    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    .commandBufferCount = cmd_count,
            };
            vk_check(vkAllocateCommandBuffers(device, &cbai, rt.command_buffers.data()));

            for (u32 i = 0; i < cmd_count; ++i) {
                set_debug_name(device, VK_OBJECT_TYPE_COMMAND_BUFFER, rt.command_buffers[i],
                               std::format("imgui_viewport_cmd_{}", i));
            }
        }

        if (rt.fences.size() != cmd_count) {
            for (auto f: rt.fences) {
                if (f != VK_NULL_HANDLE) {
                    vkDestroyFence(device, f, nullptr);
                }
            }
            rt.fences.assign(cmd_count, VK_NULL_HANDLE);

            VkFenceCreateInfo fci{
                    .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = VK_FENCE_CREATE_SIGNALED_BIT, // start signaled so first frame doesn't wait
            };
            for (u32 i = 0; i < cmd_count; ++i) {
                vk_check(vkCreateFence(device, &fci, nullptr, &rt.fences[i]));
                set_debug_name(device, VK_OBJECT_TYPE_FENCE, rt.fences[i], std::format("imgui_viewport_fence_{}", i));
            }
        }
    }

    static auto get_or_create_viewport_target(RenderContext &ctx, GLFWwindow *w) -> ImGuiViewportRenderTarget & {
        auto it = viewport_targets.find(w);
        if (it != viewport_targets.end()) {
            return it->second;
        }

        ImGuiViewportRenderTarget rt{};
        rt.window = w;

        VkSurfaceKHR surface = VK_NULL_HANDLE;
        vk_check(glfwCreateWindowSurface(ctx.get_instance(), w, nullptr, &surface));
        rt.surface = surface;

        // Create swapchain
        int fb_w = 0, fb_h = 0;
        glfwGetFramebufferSize(w, &fb_w, &fb_h);

        SwapchainCreateInfo sci{
                .physical_device = ctx.get_physical_device(),
                .device = ctx.get_device(),
                .surface = rt.surface,
                .graphics_family = ctx.queues.graphics.family_index,
                .extent = VkExtent2D{.width = static_cast<u32>(std::max(fb_w, 1)),
                                     .height = static_cast<u32>(std::max(fb_h, 1))},
                .vsync = true,
                .preferred_format = VK_FORMAT_B8G8R8A8_SRGB,
                .preferred_color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
        };

        auto sc = Swapchain::create(sci);
        if (!sc) {
            rt.swapchain = Swapchain{};
        } else {
            rt.swapchain = std::move(*sc);
        }

        ensure_viewport_command_resources(ctx, rt, static_cast<u32>(rt.swapchain.image_count()));

        auto [ins_it, _] = viewport_targets.emplace(w, std::move(rt));
        return ins_it->second;
    }

} // namespace

ImGuiRenderer::ImGuiRenderer(GLFWwindow *w, u32 initial_slot_count, RenderContext &c, Compiler &comp, FontChoice font) : ctx(c), compiler(comp) {

    std::ignore = ImGui::CreateContext();
    std::ignore = ImPlot::CreateContext();

    ImGuiIO &io = ImGui::GetIO();
    io.BackendRendererName = "imgui-custom-vulkan";
    io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;

    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        io.BackendFlags |= ImGuiBackendFlags_PlatformHasViewports;
        io.BackendFlags |= ImGuiBackendFlags_RendererHasViewports;

        ImGuiStyle &style = ImGui::GetStyle();
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    update_font(std::move(font));

    ImGui_ImplGlfw_InitForVulkan(w, true);

    slots_per_frame = std::max(1u, initial_slot_count);

    drawables.resize(frames_in_flight * slots_per_frame);
}

ImGuiRenderer::~ImGuiRenderer() {
    ImGuiIO &io = ImGui::GetIO();
    io.Fonts->TexID = nullptr;

    for (auto &[win, rt]: viewport_targets) {
        destroy_viewport_target(ctx, rt);
    }
    viewport_targets.clear();

    ImGui_ImplGlfw_Shutdown();

    ImGui::DestroyPlatformWindows();

    ImPlot::DestroyContext();
    ImGui::DestroyContext();
}

auto ImGuiRenderer::begin_frame(ImGuiFramebuffer fb) -> void {
    const auto &dim = std::get<VkExtent2D>(fb);

    ImGuiIO &io = ImGui::GetIO();
    io.DisplaySize = ImVec2(dim.width / display_scale, dim.height / display_scale);
    io.DisplayFramebufferScale = ImVec2(display_scale, display_scale);
    if (std::filesystem::create_directory("assets/editor")) {
        // ensure the ini file exists so ImGui doesn't error when trying to write to it later
        std::ofstream ini_file("assets/editor/imgui.ini");
    }
    io.IniFilename = "assets/editor/imgui.ini";

    if (force_recompile_primary || main_pipeline.empty()) {
        auto created = create_pipeline(std::get<1>(fb)).value();
        main_pipeline = Holder{ctx, ctx.create_pipeline(std::move(created))};
        force_recompile_primary = false;
    }

    if (force_recompile_offscreen || offscreen_target_pipeline.empty()) {
        auto created = create_pipeline(std::get<2>(fb)).value();
        offscreen_target_pipeline = Holder{ctx, ctx.create_pipeline(std::move(created))};
        force_recompile_offscreen = false;
    }

    slot_cursor = 0;

    frame_cursor = (frame_cursor + 1) % frames_in_flight;

    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

auto ImGuiRenderer::acquire_draw_slot() -> DrawableData & {
    if (slot_cursor >= slots_per_frame) {
        u32 new_slots_per_frame = std::max(slots_per_frame * 2u, slot_cursor + 1u);
        std::vector<DrawableData> new_drawables(frames_in_flight * new_slots_per_frame);

        for (u32 f = 0; f < frames_in_flight; ++f) {
            for (u32 s = 0; s < slots_per_frame; ++s) {
                new_drawables[f * new_slots_per_frame + s] = std::move(drawables[f * slots_per_frame + s]);
            }
        }

        drawables = std::move(new_drawables);
        slots_per_frame = new_slots_per_frame;
    }

    DrawableData &out = drawables[frame_cursor * slots_per_frame + slot_cursor];
    slot_cursor++;
    return out;
}

auto ImGuiRenderer::end_frame() -> void {
    ImGui::EndFrame();
    ImGui::Render();

    if (auto &io = ImGui::GetIO(); io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        ImGui::UpdatePlatformWindows();
    }

#ifdef NDEBUG
    frame_was_ended = true;
#endif
}

auto ImGuiRenderer::render(VkCommandBuffer cmd) -> void {
#ifdef NDEBUG
    assert(frame_was_ended && "Must call end_frame before render");
    frame_was_ended = false;
#endif

    render_draw_data(cmd, ImGui::GetDrawData(), main_pipeline);

    if (auto &io = ImGui::GetIO(); io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        render_additional_viewports();
    }
}


constexpr std::size_t next_power_of_two(std::size_t n) {
    if (n == 0) return 1;
    return std::bit_ceil(n);
}

auto ImGuiRenderer::render_draw_data(VkCommandBuffer cmd, ImDrawData *dd, PipelineHandle pipeline) -> void {
    if (!dd || dd->TotalIdxCount == 0) {
        return;
    }

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

    DrawableData &drawable = acquire_draw_slot();

    if (static_cast<i32>(drawable.index_count) < dd->TotalIdxCount) {
        const auto size = (dd->TotalIdxCount * 4) * sizeof(ImDrawIdx);
        const auto actual_size = static_cast<std::size_t>(next_power_of_two(size));
        info("(ImGui) Reallocating index buffer to {} bytes", actual_size);
        auto buffer = Buffer::zeroes(ctx.allocator, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                     actual_size, "imgui_index");
        drawable.index = Holder{ctx, ctx.create_buffer(std::move(buffer.value()))};
        drawable.index_count = static_cast<u32>(actual_size / sizeof(ImDrawIdx));
    }
    if (static_cast<i32>(drawable.vertex_count) < dd->TotalVtxCount) {
        const auto size = (dd->TotalVtxCount * 4) * sizeof(ImDrawVert);
        const auto actual_size = static_cast<std::size_t>(next_power_of_two(size));
        info("(ImGui) Reallocating vertex buffer to {} bytes", actual_size);
        auto buffer = Buffer::zeroes(ctx.allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     actual_size, "imgui_vertex");
        drawable.vertex = Holder{ctx, ctx.create_buffer(std::move(buffer.value()))};
        drawable.vertex_count = static_cast<u32>(actual_size / sizeof(ImDrawVert));
    }

    {
        std::vector<ImDrawVert> all_vtx;
        std::vector<ImDrawIdx> all_itx;

        all_vtx.reserve(static_cast<std::size_t>(dd->TotalVtxCount));
        all_itx.reserve(static_cast<std::size_t>(dd->TotalIdxCount));

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

            clipMin.x = std::max(clipMin.x, 0.0f);
            clipMin.y = std::max(clipMin.y, 0.0f);
            clipMax.x = std::min(clipMax.x, fb_width);
            clipMax.y = std::min(clipMax.y, fb_height);

            if (clipMax.x <= clipMin.x || clipMax.y <= clipMin.y) {
                continue;
            }

            struct VulkanImguiBindData {
                std::array<float, 4> LRTB{};
                const DeviceAddress vb;
                u32 base_vertex;
                u32 texture_id{0};
                u32 sampler_id{0};
            } bindData{
                    .LRTB = {L, R, T, B},
                    .vb = ctx.device_address(drawable.vertex),
                    .base_vertex = vertex_offset + imgui_cmd.VtxOffset,
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

auto ImGuiRenderer::render_additional_viewports() -> void {
    ImGuiPlatformIO &pio = ImGui::GetPlatformIO();

    for (int i = 0; i < pio.Viewports.Size; ++i) {
        ImGuiViewport *vp = pio.Viewports[i];

        if (!(vp->Flags & ImGuiViewportFlags_IsPlatformWindow)) {
            continue;
        }

        if (vp == ImGui::GetMainViewport()) {
            continue;
        }

        auto *glfw_win = static_cast<GLFWwindow *>(vp->PlatformHandle);
        if (!glfw_win || !vp->DrawData) {
            continue;
        }

        if (vp->PlatformRequestClose) {
            auto it = viewport_targets.find(glfw_win);
            if (it != viewport_targets.end()) {
                vkDeviceWaitIdle(ctx.get_device());
                destroy_viewport_target(ctx, it->second);
                viewport_targets.erase(it);
            }
            continue;
        }

        VkExtent2D want_extent{
                .width = static_cast<u32>(std::max(0.0f, vp->Size.x)),
                .height = static_cast<u32>(std::max(0.0f, vp->Size.y)),
        };
        if (want_extent.width == 0 || want_extent.height == 0) {
            continue;
        }

        auto &rt = get_or_create_viewport_target(ctx, glfw_win);
        if (rt.swapchain.image_count() == 0) {
            continue;
        }

        if (rt.swapchain.extent().width != want_extent.width || rt.swapchain.extent().height != want_extent.height) {
            auto rec = rt.swapchain.recreate(want_extent);
            if (!rec) {
                continue;
            }
            ensure_viewport_command_resources(ctx, rt, static_cast<u32>(rt.swapchain.image_count()));
        }

        const u32 image_index = (rt.frame_index - 1) % rt.swapchain.image_count();

        VkFence fence = rt.fences[image_index];
        vk_check(vkWaitForFences(ctx.get_device(), 1, &fence, VK_TRUE, UINT64_MAX));
        vk_check(vkResetFences(ctx.get_device(), 1, &fence));

        auto acq = rt.swapchain.acquire_next_image(rt.frame_index);
        if (!acq) {
            continue;
        }

        const u32 acquired_image_index = acq->image_index;

        VkCommandBuffer cmd = rt.command_buffers[acquired_image_index];
        vk_check(vkResetCommandBuffer(cmd, 0));

        VkCommandBufferBeginInfo cbbi{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .pNext = nullptr,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                .pInheritanceInfo = nullptr,
        };
        vk_check(vkBeginCommandBuffer(cmd, &cbbi));

        VkImage img = rt.swapchain.image(acquired_image_index);

        VkImageMemoryBarrier2 to_color{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .pNext = nullptr,
                .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                .srcAccessMask = VK_ACCESS_2_NONE,
                .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = img,
                .subresourceRange =
                        {
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .baseMipLevel = 0,
                                .levelCount = 1,
                                .baseArrayLayer = 0,
                                .layerCount = 1,
                        },
        };

        VkDependencyInfo dep_to_color{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pNext = nullptr,
                .dependencyFlags = 0,
                .memoryBarrierCount = 0,
                .pMemoryBarriers = nullptr,
                .bufferMemoryBarrierCount = 0,
                .pBufferMemoryBarriers = nullptr,
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &to_color,
        };
        vkCmdPipelineBarrier2(cmd, &dep_to_color);

        VkRenderingAttachmentInfo color_attachment{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = nullptr,
                .imageView = rt.swapchain.image_view(acquired_image_index),
                .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .resolveMode = VK_RESOLVE_MODE_NONE,
                .resolveImageView = VK_NULL_HANDLE,
                .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue = VkClearValue{.color = {{0.0f, 0.0f, 0.0f, 1.0f}}},
        };

        VkRenderingInfo ri{
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .pNext = nullptr,
                .flags = 0,
                .renderArea = {.offset = {0, 0}, .extent = rt.swapchain.extent()},
                .layerCount = 1,
                .viewMask = 0,
                .colorAttachmentCount = 1,
                .pColorAttachments = &color_attachment,
                .pDepthAttachment = nullptr,
                .pStencilAttachment = nullptr,
        };

        vkCmdBeginRendering(cmd, &ri);

        render_draw_data(cmd, vp->DrawData, offscreen_target_pipeline);

        vkCmdEndRendering(cmd);

        VkImageMemoryBarrier2 to_present{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .pNext = nullptr,
                .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_NONE,
                .dstAccessMask = VK_ACCESS_2_NONE,
                .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = img,
                .subresourceRange =
                        {
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .baseMipLevel = 0,
                                .levelCount = 1,
                                .baseArrayLayer = 0,
                                .layerCount = 1,
                        },
        };

        VkDependencyInfo dep_to_present{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pNext = nullptr,
                .dependencyFlags = 0,
                .memoryBarrierCount = 0,
                .pMemoryBarriers = nullptr,
                .bufferMemoryBarrierCount = 0,
                .pBufferMemoryBarriers = nullptr,
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &to_present,
        };
        vkCmdPipelineBarrier2(cmd, &dep_to_present);

        vk_check(vkEndCommandBuffer(cmd));

        VkSemaphore wait_sem = acq->sync.image_available;
        VkSemaphore signal_sem = acq->sync.render_finished;

        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        VkSubmitInfo si{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = nullptr,
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &wait_sem,
                .pWaitDstStageMask = &wait_stage,
                .commandBufferCount = 1,
                .pCommandBuffers = &cmd,
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = &signal_sem,
        };

        vk_check(vkQueueSubmit(ctx.queues.graphics.queue, 1, &si, fence));

        rt.swapchain.present(ctx.queues.graphics.queue, acquired_image_index, signal_sem);

        rt.frame_index++;
    }
}

auto ImGuiRenderer::create_pipeline(VkFormat fb) -> tl::expected<CompiledPipeline, Error> {

    constexpr std::array<const std::string_view, 2> entry_points{"vs_main", "fs_main"};
    std::array<ReflectionData, entry_points.size()> reflection{};
    TRY_PROPAGATE(shaders, compiler.compile_from_file("shaders/gui.slang", entry_points, reflection),
                  "Could not compile gui shader");

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

    struct PC {
        glm::vec4 LRTB;
        const DeviceAddress vertices;
        u32 base_vertex;
        u32 texture_id;
        u32 sampler_id;
    };


    VkPushConstantRange push_constant_range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(PC),
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
    VkFormat color_format = fb;
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
                                               ctx.allocator, *ctx.command_ctx, width, height, VK_FORMAT_R8G8B8A8_UNORM,
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
