// BaseApplication.cxx
#include "framework/BaseApplication.hxx"

#include <GLFW/glfw3.h>
#include <chrono>
#include <csignal>
#include <limits>
#include <thread>

#include "ArgumentParse.hxx"
#include "BindlessHeadless.hxx"
#include "BindlessSet.hxx"
#include "Compiler.hxx"
#include "Constants.hxx"
#include "GlobalCommandContext.hxx"
#include "ImGuiRenderer.hxx"
#include "Logger.hxx"
#include "Pipelines.hxx"
#include "Swapchain.hxx"
#include "Types.hxx"

static volatile sig_atomic_t g_keep_running = 1;
static void sig_handler(int) { g_keep_running = 1; }

struct BaseApplication::Impl {
    CLIOptions opts{};
    InstanceWithDebug *instance{nullptr};

    u32 graphics_family{0};
    VkQueue graphics_queue{VK_NULL_HANDLE};

    VkSurfaceKHR surface{VK_NULL_HANDLE};
    GraphicsTimeline tl_graphics{};
    TransferTimeline tl_transfer{};
    BindlessSet bindless{};
    Swapchain swapchain{};
    RenderContext ctx{};

    std::unique_ptr<Compiler> compiler{};
    std::unique_ptr<ImGuiRenderer> gui{};

    struct Frame {
        VkCommandPool pool{VK_NULL_HANDLE};
        VkCommandBuffer cmd{VK_NULL_HANDLE};
        u64 frame_done_value{0};
    };
    std::array<Frame, frames_in_flight> frames{};

    std::chrono::high_resolution_clock::time_point last_frame_time{};
};

namespace {
    auto validation_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
                                   const VkDebugUtilsMessengerCallbackDataEXT *data, void *) -> VkBool32 {
        if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
            std::string obj_info;
            if (data->objectCount > 0) {
                obj_info += "Objects involved:\n";
                for (u32 i = 0; i < data->objectCount; ++i) {
                    const auto &obj = data->pObjects[i];
                    if (obj.pObjectName)
                        obj_info += std::format("    Name: {}\n", obj.pObjectName);
                    obj_info += std::format("  - Object {}: Type={}, Handle=0x{:X}\n", i,
                                            static_cast<i32>(obj.objectType), obj.objectHandle);
                }
            }
            error("Validation layer: {}. {}", data->pMessage, obj_info);
        }
        return VK_FALSE;
    }
} // namespace

auto BaseApplication::on_init() -> tl::expected<void, Error> {
    auto &impl = *this->m_impl;
    {
        std::array<u8, 4> white{255, 255, 255, 255};
        std::array<u8, 4> black{0, 0, 0, 255};
        std::array<u8, 4> flat{128, 128, 255, 255};
        impl.ctx.create_texture(create_image_from_span_v2(impl.ctx.allocator, *impl.ctx.command_ctx, 1, 1,
                                                          VK_FORMAT_R8G8B8A8_UNORM, std::as_bytes(std::span(white)),
                                                          "white"));
        impl.ctx.create_texture(create_image_from_span_v2(impl.ctx.allocator, *impl.ctx.command_ctx, 1, 1,
                                                          VK_FORMAT_R8G8B8A8_UNORM, std::as_bytes(std::span(black)),
                                                          "black"));
        impl.ctx.create_texture(create_image_from_span_v2(impl.ctx.allocator, *impl.ctx.command_ctx, 1, 1,
                                                          VK_FORMAT_R8G8B8A8_UNORM, std::as_bytes(std::span(flat)),
                                                          "flat_normal"));
    }

    {
        {
            auto ci = create_info<VkSamplerCreateInfo>();
            ci.magFilter = VK_FILTER_LINEAR;
            ci.minFilter = VK_FILTER_LINEAR;
            ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            ci.maxAnisotropy = 16.0f;
            ci.anisotropyEnable = VK_TRUE;
            ci.maxLod = VK_LOD_CLAMP_NONE;
            ci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

            impl.ctx.create_sampler(ci, "linear_repeat");
        }

        {
            auto ci = create_info<VkSamplerCreateInfo>();
            ci.magFilter = VK_FILTER_LINEAR;
            ci.minFilter = VK_FILTER_LINEAR;
            ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            ci.compareOp = VK_COMPARE_OP_ALWAYS;
            ci.maxLod = VK_LOD_CLAMP_NONE;
            ci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

            impl.ctx.create_sampler(ci, "linear_clamp");
        }

        {
            auto ci = create_info<VkSamplerCreateInfo>();
            ci.magFilter = VK_FILTER_LINEAR;
            ci.minFilter = VK_FILTER_LINEAR;
            ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            ci.minLod = 0.0f;
            ci.maxLod = VK_LOD_CLAMP_NONE;
            ci.maxAnisotropy = 16.0f;
            ci.anisotropyEnable = VK_TRUE;

            impl.ctx.create_sampler(ci, "noise_sampler");
        }

        {
            auto ci = create_info<VkSamplerCreateInfo>();
            ci.magFilter = VK_FILTER_LINEAR;
            ci.minFilter = VK_FILTER_LINEAR;
            ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
            ci.compareEnable = VK_TRUE;
            ci.compareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
            ci.minLod = 0.0f;
            ci.maxLod = 0.0f;
            ci.anisotropyEnable = VK_FALSE;
            ci.unnormalizedCoordinates = VK_FALSE;

            impl.ctx.create_comparison_sampler(ci, "depth_compare_filter");
        }
    }
    impl.ctx.bindless_set->repopulate_if_needed(impl.ctx.textures, impl.ctx.samplers, impl.ctx.comparison_samplers);

    return {};
}

auto run_application(BaseApplication &app, int argc, char **argv) -> int {
    signal(SIGINT, sig_handler);

    if (glfwPlatformSupported(GLFW_PLATFORM_X11))
        glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);

    if (glfwInit() != GLFW_TRUE) {
        error("Could not initialize GLFW");
        return 1;
    }

    volkInitialize();

    app.m_impl = std::unique_ptr<BaseApplication::Impl, BaseApplication::ImplDeleter>(new BaseApplication::Impl{},
                                                                                      BaseApplication::ImplDeleter{});
    auto &impl = *app.m_impl;

    auto maybe_opts = parse_cli(argc, argv);
    if (!maybe_opts)
        return 1;
    impl.opts = std::move(maybe_opts.value());

    impl.compiler = std::make_unique<Compiler>();

    const bool enable_validation = impl.opts.validation_layers.value_or(!static_cast<bool>(IS_RELEASE));

    u32 ext_count{};
    const char **exts_raw = glfwGetRequiredInstanceExtensions(&ext_count);
    std::vector<std::string_view> extensions(exts_raw, exts_raw + ext_count);

    InstanceWithDebug instance;
    if (enable_validation)
        instance = create_instance_with_debug(validation_debug_callback, extensions);
    else {
        instance.instance = create_instance(extensions);
        instance.messenger = VK_NULL_HANDLE;
    }

    impl.instance = &instance;

    if (auto r = app.init_vulkan(impl.opts, instance); !r) {
        error("BaseApplication: init_vulkan failed: {}", r.error().message);
        return 1;
    }

    if (auto r = app.on_init(); !r) {
        error("BaseApplication: on_init failed: {}", r.error().message);
        return 1;
    }

    impl.last_frame_time = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(app.window) && g_keep_running) {
        glfwPollEvents();

        const auto now = std::chrono::high_resolution_clock::now();
        const float dt = std::chrono::duration<float>(now - impl.last_frame_time).count();
        impl.last_frame_time = now;

        const u32 fi = static_cast<u32>(app.frame_index % frames_in_flight);
        app.wait_frame(fi);

        if (auto r = app.render_frame(fi, dt); !r) {
            error("BaseApplication: render_frame failed: {}", r.error().message);
            break;
        }

        app.frame_index++;
    }

    app.on_shutdown();

    vkDeviceWaitIdle(app.device);
    impl.gui.reset();
    impl.ctx.clear_all();
    impl.ctx.destroy_queue.retire(std::numeric_limits<u64>::max());

    for (auto &f: impl.frames)
        if (f.pool != VK_NULL_HANDLE)
            vkDestroyCommandPool(app.device, f.pool, nullptr);

    destruction::global_command_context(*impl.ctx.command_ctx);
    destruction::bindless_set(impl.bindless);
    destruction::timelines(app.device, impl.tl_graphics, impl.tl_transfer, impl.tl_transfer);
    destruction::allocator(impl.ctx.allocator);
    destruction::swapchain(impl.swapchain);
    destruction::wsi(instance.instance, impl.surface, app.window);
    destruction::device(app.device);
    destruction::instance(instance);
    volkFinalize();
    glfwTerminate();

    return 0;
}

BaseApplication::BaseApplication() = default;
BaseApplication::~BaseApplication() = default;

auto BaseApplication::ImplDeleter::operator()(Impl *ptr) const noexcept -> void { delete ptr; }

auto BaseApplication::init_vulkan(CLIOptions &opts, InstanceWithDebug &instance) -> tl::expected<void, Error> {
    auto &impl = *m_impl;

    auto chosen = pick_physical_device(instance.instance);
    if (!chosen)
        return tl::unexpected(Error::make_error(Error::Type::DeviceSelectionError, "No suitable GPU"));

    auto &&[phys, gfx_i, comp_i, xfer_i] = *chosen;
    physical_device = phys;
    impl.graphics_family = gfx_i;

    auto &&[dev, gfx_q, comp_q, xfer_q, enabled] = create_device(phys, gfx_i, comp_i, xfer_i);
    device = dev;
    impl.graphics_queue = gfx_q;

    auto title = opts.title.has_value() ? opts.title.value() : std::string{};
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(static_cast<i32>(opts.width), static_cast<i32>(opts.height),
                              opts.title.has_value() ? title.c_str() : "Tool", nullptr, nullptr);
    if (!window)
        return tl::unexpected(Error::make_error(Error::Type::WindowError, "glfwCreateWindow failed"));

    glfwCreateWindowSurface(instance.instance, window, nullptr, &impl.surface);

    auto swap = Swapchain::create(SwapchainCreateInfo{
            .physical_device = physical_device,
            .device = device,
            .surface = impl.surface,
            .graphics_family = impl.graphics_family,
            .extent = VkExtent2D{opts.width, opts.height},
            .vsync = opts.vsync,
    });
    if (!swap)
        return tl::unexpected(Error::make_error(Error::Type::SwapchainError, "Swapchain create failed"));
    impl.swapchain = std::move(swap.value());

    allocator = create_allocator(instance.instance, physical_device, device, &enabled);

    impl.tl_graphics = create_graphics_timeline(device, impl.graphics_queue, impl.graphics_family);
    impl.tl_transfer = create_transfer_timeline(device, impl.graphics_queue, impl.graphics_family);

    impl.bindless.init(device, query_bindless_caps(physical_device), 4u, 4u, 4u, 4u, 0u);
    impl.bindless.grow_if_needed(64u, 16u, 8u, 4u);

    impl.ctx = RenderContext{
            .allocator = allocator,
            .bindless_set = &impl.bindless,
            .command_ctx = create_global_cmd_context(device, impl.graphics_queue, impl.graphics_family),
            .pipeline_cache = std::make_unique<PipelineCache>(device, opts.pipeline_cache_dir),
            .queues =
                    {
                            .graphics = {.queue = impl.graphics_queue, .family_index = impl.graphics_family},
                            .compute = {.queue = impl.graphics_queue, .family_index = impl.graphics_family},
                            .transfer = {.queue = impl.graphics_queue, .family_index = impl.graphics_family},
                    },
    };
    ctx = &impl.ctx;

    init_frames();

    impl.gui = std::make_unique<ImGuiRenderer>(
            window, static_cast<u32>(impl.swapchain.image_count()), impl.ctx, *impl.compiler,
            FontChoice{
                    .font_path = "assets/editor/fonts/IBM_Plex_Mono/IBMPlexMono-Regular.ttf",
                    .size = 13.0f,
            });
    gui = impl.gui.get();

    glfwSetWindowSizeCallback(window, [](GLFWwindow *, int, int) {});
    glfwShowWindow(window);
    glfwFocusWindow(window);
    return {};
}

auto BaseApplication::init_frames() -> void {
    auto &impl = *m_impl;
    for (u32 i = 0; i < frames_in_flight; ++i) {
        auto &f = impl.frames[i];

        VkCommandPoolCreateInfo cpci{
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                .queueFamilyIndex = impl.graphics_family,
        };
        vk_check(vkCreateCommandPool(device, &cpci, nullptr, &f.pool));
        set_debug_name(device, VK_OBJECT_TYPE_COMMAND_POOL, f.pool, std::format("base_app_cmd_pool_{}", i));

        VkCommandBufferAllocateInfo cbai{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool = f.pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1,
        };
        vk_check(vkAllocateCommandBuffers(device, &cbai, &f.cmd));
        set_debug_name(device, VK_OBJECT_TYPE_COMMAND_BUFFER, f.cmd, std::format("base_app_cmd_{}", i));
    }
}

auto BaseApplication::wait_frame(u32 fi) -> void {
    auto &f = m_impl->frames[fi];
    if (f.frame_done_value == 0)
        return;

    VkSemaphoreWaitInfo wi{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .semaphoreCount = 1,
            .pSemaphores = &m_impl->tl_graphics.timeline,
            .pValues = &f.frame_done_value,
    };
    vk_check(vkWaitSemaphores(device, &wi, UINT64_MAX));
}

// ---------------------------------------------------------------------------
// render_frame
// ---------------------------------------------------------------------------
auto BaseApplication::render_frame(u32 fi, float dt) -> tl::expected<void, Error> {
    auto &impl = *m_impl;
    auto &f = impl.frames[fi];
    auto &swap = impl.swapchain;
    auto &tl = impl.tl_graphics;

    auto acquired = swap.acquire_next_image(fi);
    if (!acquired) {
        if (acquired.error() == VK_ERROR_OUT_OF_DATE_KHR) {
            int w = 0, h = 0;
            glfwGetFramebufferSize(window, &w, &h);
            std::ignore = swap.recreate({static_cast<u32>(w), static_cast<u32>(h)});
            return {};
        }
        vk_check(acquired.error());
        return {};
    }

    const u32 image_idx = acquired->image_index;
    const auto &sync = acquired->sync;
    VkImage swap_image = swap.image(image_idx);
    VkImageView view = swap.image_view(image_idx);
    const VkExtent2D extent = swap.extent();

    vk_check(vkResetCommandPool(device, f.pool, 0));

    VkCommandBufferBeginInfo bi{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vk_check(vkBeginCommandBuffer(f.cmd, &bi));

    // Undefined → color attachment
    {
        auto barrier = create_info<VkImageMemoryBarrier2>();
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier.srcAccessMask = 0;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.image = swap_image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        auto dep = create_info<VkDependencyInfo>();
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(f.cmd, &dep);
    }

    impl.gui->begin_frame(ImGuiFramebuffer(extent, swap.format(), swap.format(), swap.color_space()));
    on_frame(dt);
    impl.gui->end_frame();

    VkRenderingAttachmentInfo color_att{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = view,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = {.color = {.float32 = {0.1f, 0.1f, 0.1f, 1.0f}}},
    };
    VkRenderingInfo ri{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = {{0, 0}, extent},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color_att,
    };
    vkCmdBeginRendering(f.cmd, &ri);
    impl.gui->render(f.cmd);
    vkCmdEndRendering(f.cmd);

    // Color attachment → present src
    {
        auto barrier = create_info<VkImageMemoryBarrier2>();
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        barrier.dstAccessMask = 0;
        barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.image = swap_image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        auto dep = create_info<VkDependencyInfo>();
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(f.cmd, &dep);
    }

    vk_check(vkEndCommandBuffer(f.cmd));

    const u64 signal_value = ++tl.value;
    f.frame_done_value = signal_value;

    VkSemaphoreSubmitInfo wait_bin{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = sync.image_available,
            .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
    };
    VkSemaphoreSubmitInfo sig_bin{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = sync.render_finished,
            .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
    };
    VkSemaphoreSubmitInfo sig_tl{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = tl.timeline,
            .value = signal_value,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    };
    VkCommandBufferSubmitInfo cmd_si{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .commandBuffer = f.cmd,
    };
    const std::array signals = {sig_bin, sig_tl};
    VkSubmitInfo2 si{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .waitSemaphoreInfoCount = 1,
            .pWaitSemaphoreInfos = &wait_bin,
            .commandBufferInfoCount = 1,
            .pCommandBufferInfos = &cmd_si,
            .signalSemaphoreInfoCount = static_cast<u32>(signals.size()),
            .pSignalSemaphoreInfos = signals.data(),
    };
    vk_check(vkQueueSubmit2(impl.graphics_queue, 1, &si, VK_NULL_HANDLE));

    impl.ctx.destroy_queue.retire(tl.completed);
    impl.bindless.repopulate_if_needed(impl.ctx.textures, impl.ctx.samplers, impl.ctx.comparison_samplers);

    const VkResult present_res = swap.present(impl.graphics_queue, image_idx, sync.render_finished);
    if (present_res == VK_ERROR_OUT_OF_DATE_KHR || present_res == VK_SUBOPTIMAL_KHR) {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window, &w, &h);
        std::ignore = swap.recreate({static_cast<u32>(w), static_cast<u32>(h)});
    } else {
        vk_check(present_res);
    }

    return {};
}
