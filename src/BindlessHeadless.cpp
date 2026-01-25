#include "BindlessHeadless.hxx"
#include "BindlessSet.hxx"
#include "Buffer.hxx"
#include "Compiler.hxx"
#include "GlobalCommandContext.hxx"
#include "ImageOperations.hxx"
#include "PipelineCache.hxx"
#include "Pool.hxx"
#include "Reflection.hxx"

#include "3PP/PerlinNoise.hpp"

#include <chrono>

auto vk_check(VkResult result) -> void {
    if (result != VK_SUCCESS) {
        warn("Check failed: {}", static_cast<u32>(result));
        std::abort();
    }
}

namespace {
    auto format_supports_storage_image(VkPhysicalDevice physical_device, VkFormat format, VkImageTiling tiling)
            -> bool {
        // Prefer VkFormatProperties3 (core in Vulkan 1.3, also via VK_KHR_format_feature_flags2).
        VkFormatProperties3 props3{.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3, .pNext = nullptr};
        VkFormatProperties2 props2{.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, .pNext = &props3};
        vkGetPhysicalDeviceFormatProperties2(physical_device, format, &props2);

        const VkFormatFeatureFlags2 want = VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT;

        if (tiling == VK_IMAGE_TILING_OPTIMAL) {
            return (props3.optimalTilingFeatures & want) != 0;
        }
        if (tiling == VK_IMAGE_TILING_LINEAR) {
            return (props3.linearTilingFeatures & want) != 0;
        }
        return false;
    }

    auto make_color_image_usage(VkPhysicalDevice physical_device, VkFormat format, VkSampleCountFlagBits samples,
                                bool want_sampled, bool want_storage, bool want_transfer) -> VkImageUsageFlags {
        const bool is_msaa = samples != VK_SAMPLE_COUNT_1_BIT;

        VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        if (!is_msaa) {
            if (want_sampled) {
                usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
            }

            if (want_transfer) {
                usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            }

            if (want_storage) {
                if (format_supports_storage_image(physical_device, format, VK_IMAGE_TILING_OPTIMAL)) {
                    usage |= VK_IMAGE_USAGE_STORAGE_BIT;
                }
            }
        } else {
            // Should we allow sampling from MSAA images? Nah.
            (void) want_sampled;
            (void) want_transfer;
            (void) want_storage;
        }

        return usage;
    }

    auto make_depth_image_usage(VkSampleCountFlagBits samples, bool want_sampled) -> VkImageUsageFlags {
        VkImageUsageFlags usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

        // Same logic: only sample depth if it's single-sample unless you intentionally do MSAA depth sampling.
        if (samples == VK_SAMPLE_COUNT_1_BIT && want_sampled) {
            usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
        }
        return usage;
    }

    auto choose_depth_aspect(VkFormat format) -> VkImageAspectFlags {
        VkImageAspectFlags aspect_mask = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT) {
            aspect_mask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        return aspect_mask;
    }
} // namespace

namespace destruction {
    auto instance(InstanceWithDebug const &inst) -> void {
        if (inst.instance == VK_NULL_HANDLE) {
            return;
        }

        if (inst.messenger != VK_NULL_HANDLE) {
            auto destroy_debug = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                    vkGetInstanceProcAddr(inst.instance, "vkDestroyDebugUtilsMessengerEXT"));
            if (destroy_debug) {
                destroy_debug(inst.instance, inst.messenger, nullptr);
            }
        }

        vkDestroyInstance(inst.instance, nullptr);
    }

    auto device(VkDevice &dev) -> void {
        if (dev) {
            vkDestroyDevice(dev, nullptr);
        }
        dev = VK_NULL_HANDLE;
    }

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

    auto allocator(VmaAllocator &alloc) -> void {
        if (alloc) {
            vmaDestroyAllocator(alloc);
        }
        alloc = nullptr;
    }

    auto timeline_compute(VkDevice device, ComputeTimeline &comp) -> void {
        if (comp.pool)
            vkDestroyCommandPool(device, comp.pool, nullptr);
        if (comp.timeline)
            vkDestroySemaphore(device, comp.timeline, nullptr);
        comp = {};
    }

    auto timeline_compute(VkDevice device, GraphicsTimeline &comp) -> void {
        if (comp.pool)
            vkDestroyCommandPool(device, comp.pool, nullptr);
        if (comp.timeline)
            vkDestroySemaphore(device, comp.timeline, nullptr);
        comp = {};
    }

    auto pipeline(VkDevice dev, VkPipeline &p, VkPipelineLayout &l) -> void {
        if (p != VK_NULL_HANDLE) {
            vkDestroyPipeline(dev, p, nullptr);
        }
        if (l != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(dev, l, nullptr);
        }
        p = VK_NULL_HANDLE;
        l = VK_NULL_HANDLE;
    }
} // namespace destruction

namespace detail {
    MaybeNoOp<PFN_vkSetDebugUtilsObjectNameEXT> set_debug_name_func;

    auto initialise_debug_name_func(VkInstance inst) -> void {
        auto &func = set_debug_name_func;
        if (func.empty()) {
            func = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
                    vkGetInstanceProcAddr(inst, "vkSetDebugUtilsObjectNameEXT"));
        }
    }

    auto set_debug_name_impl(VkDevice &dev, VkObjectType object_type, std::uint64_t object_handle,
                             std::string_view name) -> void {
        VkDebugUtilsObjectNameInfoEXT name_info{.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                                                .pNext = nullptr,
                                                .objectType = object_type,
                                                .objectHandle = object_handle,
                                                .pObjectName = name.data()};

        if (auto res = set_debug_name_func(dev, &name_info)) {
            vk_check(*res); // function existed; check VkResult
        }
        // else: extension not present -> no-op
    }

    auto set_debug_name_impl(VmaAllocator &alloc, VkObjectType object_type, std::uint64_t object_handle,
                             std::string_view name) -> void {
        VmaAllocatorInfo info{};
        vmaGetAllocatorInfo(alloc, &info);
        set_debug_name_impl(info.device, object_type, object_handle, name);
    }
} // namespace detail

namespace {
    template<typename TL>
    auto create_timeline(VkDevice device, VkQueue queue, u32 family_index) -> TL {
        TL t{};
        t.queue = queue;
        t.family_index = family_index;
        t.value = 0;
        t.completed = 0;

        VkSemaphoreTypeCreateInfo type_ci{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                          .pNext = nullptr,
                                          .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                                          .initialValue = 0};
        VkSemaphoreCreateInfo sci{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &type_ci, .flags = 0};
        vk_check(vkCreateSemaphore(device, &sci, nullptr, &t.timeline));

        VkCommandPoolCreateInfo pci{
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                .queueFamilyIndex = family_index,
        };
        vk_check(vkCreateCommandPool(device, &pci, nullptr, &t.pool));
        VkCommandBufferAllocateInfo cai{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                        .pNext = nullptr,
                                        .commandPool = t.pool,
                                        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                        .commandBufferCount = TL::buffered};
        vk_check(vkAllocateCommandBuffers(device, &cai, t.cmds.data()));
        return t;
    }
} // namespace

auto create_graphics_timeline(VkDevice device, VkQueue queue, u32 family_index) -> GraphicsTimeline {
    return create_timeline<GraphicsTimeline>(device, queue, family_index);
}

auto create_compute_timeline(VkDevice device, VkQueue queue, u32 family_index) -> ComputeTimeline {
    return create_timeline<ComputeTimeline>(device, queue, family_index);
}

auto create_sampler(VmaAllocator &alloc, VkSamplerCreateInfo ci, std::string_view name) -> VkSampler {
    VkSampler sampler{};
    VmaAllocatorInfo info{};
    vmaGetAllocatorInfo(alloc, &info);
    vk_check(vkCreateSampler(info.device, &ci, nullptr, &sampler));

    set_debug_name(alloc, VK_OBJECT_TYPE_SAMPLER, sampler, name);

    return sampler;
}

auto create_offscreen_target(VmaAllocator &alloc, u32 width, u32 height, VkFormat format, VkSampleCountFlagBits samples,
                             TargetSamplerConfiguration config, std::string_view name) -> OffscreenTarget {
    OffscreenTarget t{};
    t.width = width;
    t.height = height;
    t.format = format;
    VmaAllocatorInfo ai{};
    vmaGetAllocatorInfo(alloc, &ai);

    auto want_sampled = config.sampled_storage_transfer[0];
    auto want_storage = config.sampled_storage_transfer[1];
    auto want_transfer = config.sampled_storage_transfer[2];
    const VkImageUsageFlags usage =
            make_color_image_usage(ai.physicalDevice, format, samples, want_sampled, want_storage, want_transfer);

    VkImageCreateInfo ici{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = format,
            .extent = {width, height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = samples,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VmaAllocationCreateInfo aci{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    vk_check(vmaCreateImage(alloc, &ici, &aci, &t.image, &t.allocation, nullptr));


    VkImageViewCreateInfo vci{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = t.image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                           VK_COMPONENT_SWIZZLE_IDENTITY},
            .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                 .baseMipLevel = 0,
                                 .levelCount = 1,
                                 .baseArrayLayer = 0,
                                 .layerCount = 1},
    };

    // Attachment view always (because usage always contains COLOR_ATTACHMENT_BIT in make_color_image_usage()).
    vk_check(vkCreateImageView(ai.device, &vci, nullptr, &t.attachment_view));

    // Sampled view only if legal/declared.
    if ((usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0) {
        vk_check(vkCreateImageView(ai.device, &vci, nullptr, &t.sampled_view));
        set_debug_name(alloc, VK_OBJECT_TYPE_IMAGE_VIEW, t.sampled_view, std::format("{}_sampled_view", name));
    }

    // Storage view only if legal/declared.
    if ((usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0) {
        vk_check(vkCreateImageView(ai.device, &vci, nullptr, &t.storage_view));
        set_debug_name(alloc, VK_OBJECT_TYPE_IMAGE_VIEW, t.storage_view, std::format("{}_storage_view", name));
    }

    set_debug_name(alloc, VK_OBJECT_TYPE_IMAGE, t.image, name);
    set_debug_name(alloc, VK_OBJECT_TYPE_IMAGE_VIEW, t.attachment_view, std::format("{}_attachment_view", name));
    vmaSetAllocationName(alloc, t.allocation, name.data());

    return t;
}

auto create_depth_target(VmaAllocator &alloc, u32 width, u32 height, VkFormat format, VkSampleCountFlagBits samples,
                         bool want_sampled, // usually true only for single-sample depth you intend to sample later
                         std::string_view name) -> OffscreenTarget {
    OffscreenTarget t{};
    t.width = width;
    t.height = height;
    t.format = format;

    const VkImageUsageFlags usage = make_depth_image_usage(samples, want_sampled);

    VkImageCreateInfo ici{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = format,
            .extent = {width, height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = samples,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VmaAllocationCreateInfo aci{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    vk_check(vmaCreateImage(alloc, &ici, &aci, &t.image, &t.allocation, nullptr));

    VmaAllocatorInfo ai{};
    vmaGetAllocatorInfo(alloc, &ai);

    const VkImageAspectFlags aspect = choose_depth_aspect(format);

    VkImageViewCreateInfo vci{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = t.image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                           VK_COMPONENT_SWIZZLE_IDENTITY},
            .subresourceRange =
                    {.aspectMask = aspect, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1},
    };

    // Attachment view always.
    vk_check(vkCreateImageView(ai.device, &vci, nullptr, &t.attachment_view));

    // Sampled view only if SAMPLED usage was requested and allowed by make_depth_image_usage.
    if ((usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0) {
        vk_check(vkCreateImageView(ai.device, &vci, nullptr, &t.sampled_view));
        set_debug_name(alloc, VK_OBJECT_TYPE_IMAGE_VIEW, t.sampled_view, std::format("{}_sampled_view", name));
    }

    t.storage_view = VK_NULL_HANDLE;

    set_debug_name(alloc, VK_OBJECT_TYPE_IMAGE, t.image, name);
    set_debug_name(alloc, VK_OBJECT_TYPE_IMAGE_VIEW, t.attachment_view, std::format("{}_attachment_view", name));
    vmaSetAllocationName(alloc, t.allocation, name.data());

    return t;
}

auto create_image_from_span_v2(VmaAllocator alloc, GlobalCommandContext &cmd_ctx, std::uint32_t width,
                               std::uint32_t height, VkFormat format, std::span<const std::uint8_t> data,
                               std::string_view name) -> OffscreenTarget {
    auto t = create_offscreen_target(alloc, width, height, format, VK_SAMPLE_COUNT_1_BIT, {}, name);

    if (data.empty()) {
        return t;
    }

    VmaAllocatorInfo info{};
    vmaGetAllocatorInfo(alloc, &info);

    VmaAllocationCreateInfo staging_aci{};
    staging_aci.usage = VMA_MEMORY_USAGE_AUTO;
    staging_aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    auto sz = static_cast<std::size_t>(data.size_bytes());

    VkBufferCreateInfo bci{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                           .pNext = nullptr,
                           .flags = 0,
                           .size = sz,
                           .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                           .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                           .queueFamilyIndexCount = 0,
                           .pQueueFamilyIndices = nullptr};

    VkBuffer staging{};
    VmaAllocation staging_alloc{};
    vk_check(vmaCreateBuffer(alloc, &bci, &staging_aci, &staging, &staging_alloc, nullptr));

    void *mapped{};
    vk_check(vmaMapMemory(alloc, staging_alloc, &mapped));
    std::memcpy(mapped, data.data(), sz);
    vmaUnmapMemory(alloc, staging_alloc);

    auto submit_copy = [&](VkCommandBuffer cb) {
        VkImageMemoryBarrier2 pre{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                  .pNext = nullptr,
                                  .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                                  .srcAccessMask = 0,
                                  .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                  .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                  .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                  .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  .image = t.image,
                                  .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                       .baseMipLevel = 0,
                                                       .levelCount = 1,
                                                       .baseArrayLayer = 0,
                                                       .layerCount = 1}};

        VkDependencyInfo di_pre{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                .pNext = nullptr,
                                .dependencyFlags = 0,
                                .memoryBarrierCount = 0,
                                .pMemoryBarriers = nullptr,
                                .bufferMemoryBarrierCount = 0,
                                .pBufferMemoryBarriers = nullptr,
                                .imageMemoryBarrierCount = 1,
                                .pImageMemoryBarriers = &pre};

        vkCmdPipelineBarrier2(cb, &di_pre);

        VkBufferImageCopy bic{.bufferOffset = 0,
                              .bufferRowLength = 0,
                              .bufferImageHeight = 0,
                              .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                   .mipLevel = 0,
                                                   .baseArrayLayer = 0,
                                                   .layerCount = 1},
                              .imageOffset = {0, 0, 0},
                              .imageExtent = {width, height, 1}};

        vkCmdCopyBufferToImage(cb, staging, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);

        VkImageMemoryBarrier2 post{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                   .pNext = nullptr,
                                   .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                   .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                   .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                   .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                   .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   .newLayout = VK_IMAGE_LAYOUT_GENERAL, // We always use GENERAL. This is desktop safe.
                                   .image = t.image,
                                   .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                        .baseMipLevel = 0,
                                                        .levelCount = 1,
                                                        .baseArrayLayer = 0,
                                                        .layerCount = 1}};

        VkDependencyInfo di_post{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                 .pNext = nullptr,
                                 .dependencyFlags = 0,
                                 .memoryBarrierCount = 0,
                                 .pMemoryBarriers = nullptr,
                                 .bufferMemoryBarrierCount = 0,
                                 .pBufferMemoryBarriers = nullptr,
                                 .imageMemoryBarrierCount = 1,
                                 .pImageMemoryBarriers = &post};

        vkCmdPipelineBarrier2(cb, &di_post);
    };

    // Submit and wait immediately for this operation
    submit_one_time_cmd(cmd_ctx, submit_copy, true);

    t.initialized = true;

    vmaDestroyBuffer(alloc, staging, staging_alloc);

    return t;
}

auto pick_physical_device(VkInstance instance) -> tl::expected<DeviceChoice, PhysicalDeviceChoice> {
    u32 count{};
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0u) {
        return tl::unexpected(PhysicalDeviceChoice{PhysicalDeviceChoice::Error::NoDevicesFound});
    }

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    for (VkPhysicalDevice pd: devices) {
        u32 qcount{};
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, nullptr);
        if (qcount == 0u) {
            continue;
        }

        std::vector<VkQueueFamilyProperties> qprops(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, qprops.data());

        std::optional<u32> graphics{};
        std::optional<u32> compute_dedicated{};
        std::optional<u32> compute_shared{};

        for (u32 i = 0u; i < qcount; ++i) {
            VkQueueFlags flags = qprops[i].queueFlags;

            if (flags & VK_QUEUE_GRAPHICS_BIT) {
                if (!graphics) {
                    graphics = i;
                }
                if (flags & VK_QUEUE_COMPUTE_BIT) {
                    if (!compute_shared) {
                        compute_shared = i;
                    }
                }
                continue;
            }

            if (flags & VK_QUEUE_COMPUTE_BIT) {
                if (!(flags & VK_QUEUE_GRAPHICS_BIT)) {
                    compute_dedicated = i;
                }
            }
        }

        if (graphics && compute_dedicated) {
            return DeviceChoice{pd, *graphics, *compute_dedicated};
        }

        if (graphics && compute_shared) {
            return DeviceChoice{pd, *graphics, *compute_shared};
        }
    }

    return tl::unexpected(PhysicalDeviceChoice{PhysicalDeviceChoice::Error::NoQueuesFound});
}

auto create_device(VkPhysicalDevice pd, u32 graphics_index, u32 compute_index)
        -> std::tuple<VkDevice, VkQueue, VkQueue> {
    u32 ext_count{};
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> dev_exts(ext_count);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &ext_count, dev_exts.data());

    auto has_ext = [&](char const *name) -> bool {
        for (auto const &e: dev_exts) {
            if (std::strcmp(e.extensionName, name) == 0) {
                return true;
            }
        }
        return false;
    };

    bool accel_supported = has_ext(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
                           has_ext(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

    std::vector<char const *> enabled_exts;
    if (accel_supported) {
        enabled_exts.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        enabled_exts.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    }

    enabled_exts.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
    enabled_exts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    enabled_exts.push_back(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);


    VkPhysicalDeviceFragmentShadingRateFeaturesKHR shading_rate_features_khr{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR,
            .pNext = nullptr,
            .pipelineFragmentShadingRate = VK_TRUE,
            .primitiveFragmentShadingRate = VK_TRUE,
            .attachmentFragmentShadingRate = VK_TRUE};

    VkPhysicalDeviceMeshShaderFeaturesEXT mesh_features = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
            .pNext = &shading_rate_features_khr,
            .taskShader = VK_TRUE, // Optional, but recommended for culling
            .meshShader = VK_TRUE,
            .primitiveFragmentShadingRateMeshShader = VK_TRUE,
    };
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
            .pNext = &mesh_features,
    };

    VkPhysicalDeviceVulkan11Features features11{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                                                .pNext = &accel_features};

    VkPhysicalDeviceVulkan12Features features12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                                .pNext = &features11};

    VkPhysicalDeviceVulkan13Features features13{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                                .pNext = &features12};

    VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &features13,
    };


    vkGetPhysicalDeviceFeatures2(pd, &features2);
    features2.features.robustBufferAccess = VK_TRUE;

    features11.storageBuffer16BitAccess = VK_TRUE;
    features11.uniformAndStorageBuffer16BitAccess = VK_TRUE;
    features11.multiview = VK_TRUE;
    features11.multiviewGeometryShader = VK_TRUE;
    features11.multiviewTessellationShader = VK_TRUE;
    features11.variablePointersStorageBuffer = VK_TRUE;
    features11.variablePointers = VK_TRUE;
    features11.protectedMemory = VK_FALSE;
    features11.samplerYcbcrConversion = VK_TRUE;
    features11.shaderDrawParameters = VK_TRUE;

    features12.bufferDeviceAddress = VK_TRUE;
    features12.bufferDeviceAddressCaptureReplay = VK_TRUE;
    features12.descriptorIndexing = VK_TRUE;
    features12.runtimeDescriptorArray = VK_TRUE;
    features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    features12.shaderUniformBufferArrayNonUniformIndexing = VK_TRUE;
    features12.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
    features12.descriptorBindingPartiallyBound = VK_TRUE;
    features12.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;
    features12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    features12.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
    features12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    features12.descriptorBindingUniformTexelBufferUpdateAfterBind = VK_TRUE;
    features12.descriptorBindingStorageTexelBufferUpdateAfterBind = VK_TRUE;
    features12.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
    features12.timelineSemaphore = VK_TRUE;

    features13.dynamicRendering = VK_TRUE;
    features13.synchronization2 = VK_TRUE;
    features13.robustImageAccess = VK_TRUE;

    if (accel_supported) {
        accel_features.accelerationStructure = VK_TRUE;
        accel_features.descriptorBindingAccelerationStructureUpdateAfterBind = VK_TRUE;
        accel_features.accelerationStructureCaptureReplay = VK_TRUE;
    }

    float priority_graphics = 1.0f;
    VkDeviceQueueCreateInfo qci_graphics{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                         .pNext = nullptr,
                                         .flags = 0,
                                         .queueFamilyIndex = graphics_index,
                                         .queueCount = 1u,
                                         .pQueuePriorities = &priority_graphics};

    float priority_compute = 1.0f;
    VkDeviceQueueCreateInfo qci_compute{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                        .pNext = nullptr,
                                        .flags = 0,
                                        .queueFamilyIndex = compute_index,
                                        .queueCount = 1u,
                                        .pQueuePriorities = &priority_compute};

    std::array<VkDeviceQueueCreateInfo, 2> qcis{qci_graphics, qci_compute};

    u32 qci_count = 0u;
    qcis[qci_count++] = qci_graphics;
    if (compute_index != graphics_index) {
        qcis[qci_count++] = qci_compute;
    }

    VkDeviceCreateInfo dci{.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                           .pNext = &features2,
                           .flags = 0,
                           .queueCreateInfoCount = qci_count,
                           .pQueueCreateInfos = qcis.data(),
                           .enabledLayerCount = 0,
                           .ppEnabledLayerNames = nullptr,
                           .enabledExtensionCount = static_cast<u32>(enabled_exts.size()),
                           .ppEnabledExtensionNames = enabled_exts.empty() ? nullptr : enabled_exts.data(),
                           .pEnabledFeatures = nullptr};

    VkDevice device{};
    vk_check(vkCreateDevice(pd, &dci, nullptr, &device));
    volkLoadDevice(device);

    VkQueue gq{};
    vkGetDeviceQueue(device, graphics_index, 0u, &gq);

    VkQueue cq{};
    vkGetDeviceQueue(device, compute_index, 0u, &cq);

    return {device, gq, cq};
}

auto create_allocator(VkInstance instance, VkPhysicalDevice pd, VkDevice device) -> VmaAllocator {
    VmaAllocatorCreateInfo info{};
    info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    info.physicalDevice = pd;
    info.device = device;
    info.instance = instance;

    VmaVulkanFunctions vma_vulkan_func{};
    vma_vulkan_func.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vma_vulkan_func.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    info.pVulkanFunctions = &vma_vulkan_func;

    VmaAllocator alloc{};
    vmaCreateAllocator(&info, &alloc);
    return alloc;
}

constexpr u32 max_in_flight_frames = 2;

template<typename TL>
constexpr auto max_in_flight_submits() -> u64 {
    return static_cast<u64>(max_in_flight_frames) * TL::submits_per_frame;
}

auto throttle(ComputeTimeline &tl, VkDevice device) -> void {
    u64 current = 0;
    vk_check(vkGetSemaphoreCounterValue(device, tl.timeline, &current));
    tl.completed = current;

    const u64 limit = max_in_flight_submits<ComputeTimeline>();
    if (tl.value <= tl.completed + limit)
        return;

    const u64 wait_val = tl.value - limit;
    VkSemaphoreWaitInfo wi{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                           .semaphoreCount = 1,
                           .pSemaphores = &tl.timeline,
                           .pValues = &wait_val};
    vk_check(vkWaitSemaphores(device, &wi, UINT64_MAX));
    tl.completed = wait_val;
}

auto throttle(GraphicsTimeline &tl, VkDevice device) -> void {
    u64 current = 0;
    vk_check(vkGetSemaphoreCounterValue(device, tl.timeline, &current));
    tl.completed = current;

    const u64 limit = max_in_flight_submits<GraphicsTimeline>();
    if (tl.value <= tl.completed + limit)
        return;

    const u64 wait_val = tl.value - limit;
    VkSemaphoreWaitInfo wi{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                           .semaphoreCount = 1,
                           .pSemaphores = &tl.timeline,
                           .pValues = &wait_val};
    vk_check(vkWaitSemaphores(device, &wi, UINT64_MAX));
    tl.completed = wait_val;
}
