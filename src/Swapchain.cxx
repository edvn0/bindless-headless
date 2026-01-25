#include "Swapchain.hxx"

#include <algorithm>
#include <limits>

static auto clamp_u32(u32 v, u32 lo, u32 hi) -> u32 { return std::min(std::max(v, lo), hi); }

static auto choose_surface_format(std::span<const VkSurfaceFormatKHR> formats, VkFormat preferred_format,
                                  VkColorSpaceKHR preferred_color_space) -> VkSurfaceFormatKHR {
    if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED) {
        return VkSurfaceFormatKHR{.format = preferred_format, .colorSpace = preferred_color_space};
    }

    for (const auto &f: formats) {
        if (f.format == preferred_format && f.colorSpace == preferred_color_space) {
            return f;
        }
    }

    for (const auto &f: formats) {
        if ((f.format == VK_FORMAT_B8G8R8A8_SRGB || f.format == VK_FORMAT_R8G8B8A8_SRGB) &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return f;
        }
    }

    return formats[0];
}

static auto choose_present_mode(std::span<const VkPresentModeKHR> modes, bool vsync) -> VkPresentModeKHR {
    const auto has = [&](VkPresentModeKHR m) -> bool { return std::ranges::find(modes, m) != modes.end(); };

    if (vsync) {
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    if (has(VK_PRESENT_MODE_MAILBOX_KHR))
        return VK_PRESENT_MODE_MAILBOX_KHR;
    if (has(VK_PRESENT_MODE_IMMEDIATE_KHR))
        return VK_PRESENT_MODE_IMMEDIATE_KHR;

    return VK_PRESENT_MODE_FIFO_KHR;
}

static auto choose_extent(const VkSurfaceCapabilitiesKHR &caps, VkExtent2D requested) -> VkExtent2D {
    if (caps.currentExtent.width != std::numeric_limits<u32>::max()) {
        return caps.currentExtent;
    }

    VkExtent2D e = requested;
    e.width = clamp_u32(e.width, caps.minImageExtent.width, caps.maxImageExtent.width);
    e.height = clamp_u32(e.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    e.width = std::max(e.width, 1u);
    e.height = std::max(e.height, 1u);
    return e;
}

static auto choose_image_count(const VkSurfaceCapabilitiesKHR &caps) -> u32 {
    u32 count = caps.minImageCount + 1u;
    if (caps.maxImageCount != 0) {
        count = std::min(count, caps.maxImageCount);
    }
    return count;
}

auto Swapchain::destroy_swapchain_resources() -> void {
    for (auto v: views) {
        if (v != VK_NULL_HANDLE) {
            vkDestroyImageView(device, v, nullptr);
        }
    }
    views.clear();
    images.clear();


    for (auto &s: render_finished_semaphores) {
        if (s != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, s, nullptr);
            s = VK_NULL_HANDLE;
        }
    }
    render_finished_semaphores.clear();

    if (swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }

    surface_format = {};
    present_mode = VK_PRESENT_MODE_FIFO_KHR;
    full_extent = {};
}

auto Swapchain::destroy() -> void {
    if (device == VK_NULL_HANDLE) {
        *this = Swapchain{};
        return;
    }

    destroy_swapchain_resources();


    for (auto &s: acquire_semaphores) {
        if (s != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, s, nullptr);
            s = VK_NULL_HANDLE;
        }
    }
    acquire_semaphores.clear();

    device = VK_NULL_HANDLE;
    physical_device = VK_NULL_HANDLE;
    surface = VK_NULL_HANDLE;
    graphics_family = 0;

    vsync = true;
    preferred_format = VK_FORMAT_B8G8R8A8_SRGB;
    preferred_color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
}

auto Swapchain::create_swapchain_resources(VkExtent2D requested_extent, bool use_vsync, VkFormat want_format,
                                           VkColorSpaceKHR want_color_space, VkSwapchainKHR old_swapchain)
        -> tl::expected<void, VkResult> {
    VkSurfaceCapabilitiesKHR caps{};
    VkResult res = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &caps);
    if (res != VK_SUCCESS)
        return tl::unexpected(res);

    u32 format_count = 0;
    res = vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, nullptr);
    if (res != VK_SUCCESS)
        return tl::unexpected(res);
    if (format_count == 0)
        return tl::unexpected(VK_ERROR_INITIALIZATION_FAILED);

    std::vector<VkSurfaceFormatKHR> formats(format_count);
    res = vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, formats.data());
    if (res != VK_SUCCESS)
        return tl::unexpected(res);

    u32 present_mode_count = 0;
    res = vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_mode_count, nullptr);
    if (res != VK_SUCCESS)
        return tl::unexpected(res);
    if (present_mode_count == 0)
        return tl::unexpected(VK_ERROR_INITIALIZATION_FAILED);

    std::vector<VkPresentModeKHR> present_modes(present_mode_count);
    res = vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_mode_count,
                                                    present_modes.data());
    if (res != VK_SUCCESS)
        return tl::unexpected(res);

    surface_format = choose_surface_format(formats, want_format, want_color_space);
    present_mode = choose_present_mode(present_modes, use_vsync);
    full_extent = choose_extent(caps, requested_extent);

    const u32 image_count = choose_image_count(caps);

    VkSwapchainCreateInfoKHR sci{.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                                 .surface = surface,
                                 .minImageCount = image_count,
                                 .imageFormat = surface_format.format,
                                 .imageColorSpace = surface_format.colorSpace,
                                 .imageExtent = full_extent,
                                 .imageArrayLayers = 1,
                                 .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                 .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
                                 .queueFamilyIndexCount = 0,
                                 .pQueueFamilyIndices = nullptr,
                                 .preTransform = caps.currentTransform,
                                 .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                                 .presentMode = present_mode,
                                 .clipped = VK_TRUE,
                                 .oldSwapchain = old_swapchain};

    VkSwapchainKHR new_swapchain = VK_NULL_HANDLE;
    res = vkCreateSwapchainKHR(device, &sci, nullptr, &new_swapchain);
    if (res != VK_SUCCESS)
        return tl::unexpected(res);

    u32 swap_image_count = 0;
    res = vkGetSwapchainImagesKHR(device, new_swapchain, &swap_image_count, nullptr);
    if (res != VK_SUCCESS) {
        vkDestroySwapchainKHR(device, new_swapchain, nullptr);
        return tl::unexpected(res);
    }
    if (swap_image_count == 0) {
        vkDestroySwapchainKHR(device, new_swapchain, nullptr);
        return tl::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    }

    std::vector<VkImage> new_images(swap_image_count);
    res = vkGetSwapchainImagesKHR(device, new_swapchain, &swap_image_count, new_images.data());
    if (res != VK_SUCCESS) {
        vkDestroySwapchainKHR(device, new_swapchain, nullptr);
        return tl::unexpected(res);
    }

    std::vector<VkImageView> new_views(swap_image_count, VK_NULL_HANDLE);
    for (u32 i = 0; i < swap_image_count; ++i) {
        VkImageViewCreateInfo ivci{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                   .image = new_images[i],
                                   .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                   .format = surface_format.format,
                                   .components = VkComponentMapping{.r = VK_COMPONENT_SWIZZLE_IDENTITY,
                                                                    .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                                                                    .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                                                                    .a = VK_COMPONENT_SWIZZLE_IDENTITY},
                                   .subresourceRange = VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                                               .baseMipLevel = 0,
                                                                               .levelCount = 1,
                                                                               .baseArrayLayer = 0,
                                                                               .layerCount = 1}};

        res = vkCreateImageView(device, &ivci, nullptr, &new_views[i]);
        if (res != VK_SUCCESS) {
            for (auto v: new_views) {
                if (v != VK_NULL_HANDLE) {
                    vkDestroyImageView(device, v, nullptr);
                }
            }
            vkDestroySwapchainKHR(device, new_swapchain, nullptr);
            return tl::unexpected(res);
        }
    }


    for (auto &s: render_finished_semaphores) {
        if (s != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, s, nullptr);
        }
    }
    render_finished_semaphores.resize(swap_image_count);

    VkSemaphoreCreateInfo sem_ci{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};

    for (u32 i = 0; i < swap_image_count; ++i) {
        res = vkCreateSemaphore(device, &sem_ci, nullptr, &render_finished_semaphores[i]);
        if (res != VK_SUCCESS) {
            for (auto v: new_views) {
                if (v != VK_NULL_HANDLE) {
                    vkDestroyImageView(device, v, nullptr);
                }
            }
            vkDestroySwapchainKHR(device, new_swapchain, nullptr);
            return tl::unexpected(res);
        }
    }

    swapchain = new_swapchain;
    images = std::move(new_images);
    views = std::move(new_views);

    return {};
}

auto Swapchain::recreate(VkExtent2D new_extent) -> tl::expected<void, VkResult> {
    if (device == VK_NULL_HANDLE || physical_device == VK_NULL_HANDLE || surface == VK_NULL_HANDLE) {
        return tl::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    }

    vkDeviceWaitIdle(device);

    const VkSwapchainKHR old_swapchain = swapchain;

    for (auto v: views) {
        if (v != VK_NULL_HANDLE) {
            vkDestroyImageView(device, v, nullptr);
        }
    }
    views.clear();
    images.clear();
    swapchain = VK_NULL_HANDLE;

    auto created =
            create_swapchain_resources(new_extent, vsync, preferred_format, preferred_color_space, old_swapchain);
    if (!created) {
        if (old_swapchain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device, old_swapchain, nullptr);
        }

        surface_format = {};
        present_mode = VK_PRESENT_MODE_FIFO_KHR;
        full_extent = {};
        return created;
    }

    if (old_swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device, old_swapchain, nullptr);
    }

    return {};
}

auto Swapchain::create(const SwapchainCreateInfo &ci) -> tl::expected<Swapchain, VkResult> {
    Swapchain out{};
    out.device = ci.device;
    out.physical_device = ci.physical_device;
    out.surface = ci.surface;
    out.graphics_family = ci.graphics_family;

    out.vsync = ci.vsync;
    out.preferred_format = ci.preferred_format;
    out.preferred_color_space = ci.preferred_color_space;

    out.acquire_semaphores.resize(frames_in_flight);

    VkSemaphoreCreateInfo sem_ci{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};

    for (u32 fi = 0; fi < frames_in_flight; ++fi) {
        VkResult res = vkCreateSemaphore(ci.device, &sem_ci, nullptr, &out.acquire_semaphores[fi]);
        if (res != VK_SUCCESS) {
            out.destroy();
            return tl::unexpected(res);
        }
    }

    auto created = out.create_swapchain_resources(ci.extent, ci.vsync, ci.preferred_format, ci.preferred_color_space,
                                                  VK_NULL_HANDLE);

    if (!created) {
        const VkResult res = created.error();
        out.destroy();
        return tl::unexpected(res);
    }

    return out;
}


auto Swapchain::acquire_next_image(u32 frame_index, u64 timeout_ns) -> tl::expected<SwapchainAcquireResult, VkResult> {

    const auto bounded_frame = frame_index % frames_in_flight;
    VkSemaphore acquire_sem = acquire_semaphores[bounded_frame];

    u32 image_index{0};
    const VkResult res =
            vkAcquireNextImageKHR(device, swapchain, timeout_ns, acquire_sem, VK_NULL_HANDLE, &image_index);

    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) {
        return tl::unexpected(res);
    }

    return SwapchainAcquireResult{
            .image_index = image_index,
            .sync = SwapchainFrameSync{.image_available = acquire_sem,
                                       .render_finished = render_finished_semaphores[image_index]},
    };
}

auto Swapchain::present(VkQueue queue, u32 image_index, VkSemaphore render_finished) -> VkResult {
    if (swapchain == VK_NULL_HANDLE) {
        return VK_ERROR_OUT_OF_DATE_KHR;
    }

    VkPresentInfoKHR pi{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                        .waitSemaphoreCount = render_finished ? 1u : 0u,
                        .pWaitSemaphores = render_finished ? &render_finished : nullptr,
                        .swapchainCount = 1u,
                        .pSwapchains = &swapchain,
                        .pImageIndices = &image_index,
                        .pResults = nullptr};

    return vkQueuePresentKHR(queue, &pi);
}
