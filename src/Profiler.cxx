#include "Profiler.hxx"
#include "Logger.hxx"

auto TracyGpuContext::init_calibrated(VkInstance instance, VkPhysicalDevice physdev, VkDevice dev,
                                      const VolkInstanceTable &, const std::string_view name) -> void {
#if defined(TRACY_ENABLE)
    device = dev;

    auto get_domains = reinterpret_cast<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT>(
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT"));

    auto get_timestamps = reinterpret_cast<PFN_vkGetCalibratedTimestampsEXT>(
            vkGetDeviceProcAddr(device, "vkGetCalibratedTimestampsEXT"));

    if (get_domains && get_timestamps) {
        ctx = TracyVkContextHostCalibrated(instance, physdev, device, vkGetInstanceProcAddr, vkGetDeviceProcAddr);
    } else {
        error("This app requires host calibrated");
        std::abort();
    }

    TracyVkContextName(ctx, name.data(), static_cast<u16>(name.size()));
#else
    (void) name;
    (void) dev;
    (void) physdev;
    (void) instance;
#endif
}

auto TracyGpuContext::shutdown() -> void {
#if defined(TRACY_ENABLE)
    if (ctx)
        TracyVkDestroy(ctx);
    ctx = nullptr;

    device = VK_NULL_HANDLE;
#endif
}
