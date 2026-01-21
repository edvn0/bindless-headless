#include "Profiler.hxx"

auto TracyGpuContext::init_calibrated(VkInstance instance,
                                      VkPhysicalDevice physdev,
                                      VkDevice dev,
                                      VkQueue queue,
                                      u32 queue_family_index,
                                      const char *name) -> void {
#if defined(TRACY_ENABLE)
    device = dev;

    // Create a resettable command buffer in INITIAL state
    VkCommandPoolCreateInfo cpci{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queue_family_index,
    };
    vk_check(vkCreateCommandPool(device, &cpci, nullptr, &pool));

    VkCommandBufferAllocateInfo cbai{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    vk_check(vkAllocateCommandBuffers(device, &cbai, &init_cmd));

    auto get_domains =
            reinterpret_cast<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT>(
                vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT"));

    auto get_timestamps =
            reinterpret_cast<PFN_vkGetCalibratedTimestampsEXT>(
                vkGetDeviceProcAddr(device, "vkGetCalibratedTimestampsEXT"));

    if (get_domains && get_timestamps) {
        ctx = TracyVkContextCalibrated(physdev, device, queue, init_cmd, get_domains, get_timestamps);
    } else {
        // Fallback: still works, just less perfect CPU↔GPU alignment
        ctx = TracyVkContext(physdev, device, queue, init_cmd);
    }

    TracyVkContextName(ctx, name, static_cast<u16>(std::strlen(name)));
#else
    (void) name;
    (void) queue_family_index;
    (void) queue;
    (void) dev;
    (void) physdev;
    (void) instance;
#endif
}

auto TracyGpuContext::shutdown() -> void {
#if defined(TRACY_ENABLE)
    if (ctx) TracyVkDestroy(ctx);
    ctx = nullptr;

    if (pool) {
        vkDestroyCommandPool(device, pool, nullptr);
        pool = VK_NULL_HANDLE;
    }
    device = VK_NULL_HANDLE;
    init_cmd = VK_NULL_HANDLE;
#endif
}
