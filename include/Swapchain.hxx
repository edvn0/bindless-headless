#pragma once

#include <vector>
#include <tl/expected.hpp>

#include "Types.hxx"

struct SwapchainCreateInfo {
    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    VkSurfaceKHR surface{VK_NULL_HANDLE};

    u32 graphics_family{0};
    VkExtent2D extent{};

    bool vsync{true};
    VkFormat preferred_format{VK_FORMAT_B8G8R8A8_SRGB};
    VkColorSpaceKHR preferred_color_space{VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
};

struct SwapchainFrameSync {
    VkSemaphore image_available{VK_NULL_HANDLE};
    VkSemaphore render_finished{VK_NULL_HANDLE};
};

struct SwapchainAcquireResult {
    u32 image_index{0};
    SwapchainFrameSync sync{};
};

class Swapchain {
public:
    static auto create(const SwapchainCreateInfo &ci) -> tl::expected<Swapchain, VkResult>;

    auto destroy() -> void;

    auto recreate(VkExtent2D new_extent) -> tl::expected<void, VkResult>;

    auto acquire_next_image(u32 frame_index, u64 timeout_ns = UINT64_MAX)
            -> tl::expected<SwapchainAcquireResult, VkResult>;

    auto present(VkQueue queue, u32 image_index, VkSemaphore render_finished) -> VkResult;

    [[nodiscard]] auto extent() const -> VkExtent2D { return full_extent; }
    [[nodiscard]] auto format() const -> VkFormat { return surface_format.format; }
    [[nodiscard]] auto color_space() const -> VkColorSpaceKHR { return surface_format.colorSpace; }

    [[nodiscard]] auto image(u32 index) const -> VkImage { return images[index]; }
    [[nodiscard]] auto image_view(u32 index) const -> VkImageView { return views[index]; }

    [[nodiscard]] auto frame_sync(u32 frame_index) const -> const SwapchainFrameSync & { return sync[frame_index]; }

private:
    auto destroy_swapchain_resources() -> void;

    auto create_swapchain_resources(VkExtent2D requested_extent, bool vsync, VkFormat preferred_format,
                                    VkColorSpaceKHR preferred_color_space, VkSwapchainKHR old_swapchain)
            -> tl::expected<void, VkResult>;

    VkDevice device{VK_NULL_HANDLE};
    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    VkSurfaceKHR surface{VK_NULL_HANDLE};

    VkSwapchainKHR swapchain{VK_NULL_HANDLE};
    VkSurfaceFormatKHR surface_format{};
    VkPresentModeKHR present_mode{VK_PRESENT_MODE_FIFO_KHR};
    VkExtent2D full_extent{};

    u32 graphics_family{0};

    std::vector<VkImage> images{};
    std::vector<VkImageView> views{};
    std::vector<SwapchainFrameSync> sync{};

    std::vector<VkSemaphore> acquire_semaphores; 
std::vector<VkSemaphore> render_finished_semaphores;

    bool vsync{true};
    VkFormat preferred_format{VK_FORMAT_B8G8R8A8_UNORM};
    VkColorSpaceKHR preferred_color_space{VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
};
