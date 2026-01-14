#include "BindlessHeadless.hxx"
#include "BindlessSet.hxx"
#include "GlobalCommandContext.hxx"
#include "ImageOperations.hxx"
#include "Pool.hxx"
#include "PipelineCache.hxx"
#include "Reflection.hxx"
#include "Compiler.hxx"
#include "Buffer.hxx"

#include "3PP/PerlinNoise.hpp"

#include "3PP/renderdoc_app.h"

#include <chrono>

auto vk_check(VkResult result) -> void {
    if (result != VK_SUCCESS) {
        std::cerr << "Result: " << string_VkResult(result) << "\n";
        std::abort();
    }
}

namespace destruction {
    auto bindless_set(VkDevice device, BindlessSet &bs) -> void {
        if (bs.pool) {
            vkDestroyDescriptorPool(device, bs.pool, nullptr);
        }
        if (bs.layout) {
            vkDestroyDescriptorSetLayout(device, bs.layout, nullptr);
        }
        bs.pool = VK_NULL_HANDLE;
        bs.layout = VK_NULL_HANDLE;
        bs.set = VK_NULL_HANDLE;
    }
} // namespace destruction

namespace detail {
    auto set_debug_name_impl(VmaAllocator &alloc,
                             VkObjectType object_type,
                             std::uint64_t object_handle,
                             std::string_view name) -> void {
        VmaAllocatorInfo info{};
        vmaGetAllocatorInfo(alloc, &info);

        static PFN_vkSetDebugUtilsObjectNameEXT set_debug_name_func =
                nullptr;
        if (set_debug_name_func == nullptr) {
            set_debug_name_func =
                    reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
                        vkGetInstanceProcAddr(
                            info.instance,
                            "vkSetDebugUtilsObjectNameEXT"));
        }

        if (set_debug_name_func == nullptr) {
            return;
        }

        VkDebugUtilsObjectNameInfoEXT name_info{
            .sType =
            VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .pNext = nullptr,
            .objectType = object_type,
            .objectHandle = object_handle,
            .pObjectName = name.data()
        };
        vk_check(set_debug_name_func(info.device, &name_info));
    }
} // namespace detail
