#include "BindlessHeadless.hxx"
#include "BindlessSet.hxx"
#include "Buffer.hxx"
#include "Compiler.hxx"
#include "GlobalCommandContext.hxx"
#include "ImageOperations.hxx"
#include "Logger.hxx"
#include "PipelineCache.hxx"
#include "Pool.hxx"
#include "Reflection.hxx"
#include "Swapchain.hxx"

#include "3PP/PerlinNoise.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <optional>

auto vk_check(VkResult result) -> void {
    if (result != VK_SUCCESS) {
        warn("Check failed: {}", static_cast<u32>(result));
        std::abort();
    }
}

namespace {
    auto mip_extent(u32 base_w, u32 base_h, u32 level) -> VkExtent3D {
        return VkExtent3D{
                .width = std::max(1u, base_w >> level),
                .height = std::max(1u, base_h >> level),
                .depth = 1,
        };
    }

    auto format_supports_storage_image(VkPhysicalDevice physical_device, VkFormat format, VkImageTiling tiling)
            -> bool {
        VkFormatProperties3 props3{};
        props3.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3;
        VkFormatProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;
        props2.pNext = &props3;
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
            vkDestroyDebugUtilsMessengerEXT(inst.instance, inst.messenger, nullptr);
        }

        auto *destroy = vkDestroyInstance;
        destroy(inst.instance, nullptr);
    }

    auto device(VkDevice &dev) -> void {
        if (dev) {
            vkDestroyDevice(dev, nullptr);
        }
        dev = VK_NULL_HANDLE;
    }

    auto bindless_set(BindlessSet &bs) -> void { bs.destroy(); }

    auto allocator(VmaAllocator &alloc) -> void {
        if (alloc) {
            vmaDestroyAllocator(alloc);
        }
        alloc = nullptr;
    }

    auto swapchain(Swapchain &sc) -> void { sc.destroy(); }

    auto tl(VkDevice dev, auto &t) -> void {
        if (t.pool)
            vkDestroyCommandPool(dev, t.pool, nullptr);
        if (t.timeline)
            vkDestroySemaphore(dev, t.timeline, nullptr);
        t = {};
    }

    auto timeline(VkDevice device, ComputeTimeline &t) -> void { tl(device, t); }

    auto timeline(VkDevice device, GraphicsTimeline &t) -> void { tl(device, t); }

    auto timeline(VkDevice device, TransferTimeline &t) -> void { tl(device, t); }

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

    auto set_debug_name_impl(VkDevice dev, VkObjectType object_type, std::uint64_t object_handle, std::string_view name)
            -> void {
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
    auto create_timeline(VkDevice device, VkQueue queue, u32 family_index, const std::string_view name) -> TL {
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
        set_debug_name(device, VK_OBJECT_TYPE_SEMAPHORE, t.timeline, std::format("{}_timeline", name));


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


auto create_transfer_timeline(VkDevice device, VkQueue queue, u32 family_index) -> TransferTimeline {
    return create_timeline<TransferTimeline>(device, queue, family_index, "transfer");
}

auto create_graphics_timeline(VkDevice device, VkQueue queue, u32 family_index) -> GraphicsTimeline {
    return create_timeline<GraphicsTimeline>(device, queue, family_index, "graphics");
}

auto create_compute_timeline(VkDevice device, VkQueue queue, u32 family_index) -> ComputeTimeline {
    return create_timeline<ComputeTimeline>(device, queue, family_index, "compute");
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

    VkImageUsageFlags const usage =
            make_color_image_usage(ai.physicalDevice, format, samples, want_sampled, want_storage, want_transfer);

    u32 const mip_levels = std::max(1u, config.dims.mip_levels);
    u32 const array_layers = std::max(1u, config.dims.array_layers);

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {width, height, 1};
    ici.mipLevels = mip_levels;
    ici.arrayLayers = array_layers;
    ici.samples = samples;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (array_layers > 1) {
        // Not strictly required for basic 2D array views, but it’s a helpful “this is an array image” hint.
        ici.flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    }

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(alloc, &ici, &aci, &t.image, &t.allocation, nullptr));

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = t.image;
    vci.viewType = (array_layers > 1) ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    if (config.dims.view_type != VK_IMAGE_VIEW_TYPE_2D) {
        // Allow explicit override if you want it later.
        vci.viewType = config.dims.view_type;
    }
    vci.format = format;
    vci.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                      VK_COMPONENT_SWIZZLE_IDENTITY};
    vci.subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = mip_levels,
            .baseArrayLayer = 0,
            .layerCount = array_layers,
    };

    // Attachment view
    vk_check(vkCreateImageView(ai.device, &vci, nullptr, &t.attachment_view));

    if ((usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0) {
        vk_check(vkCreateImageView(ai.device, &vci, nullptr, &t.sampled_view));
        set_debug_name(alloc, VK_OBJECT_TYPE_IMAGE_VIEW, t.sampled_view, std::format("{}_sampled_view", name));
    }

    if ((usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0) {
        vk_check(vkCreateImageView(ai.device, &vci, nullptr, &t.storage_view));
        set_debug_name(alloc, VK_OBJECT_TYPE_IMAGE_VIEW, t.storage_view, std::format("{}_storage_view", name));
    }

    set_debug_name(alloc, VK_OBJECT_TYPE_IMAGE, t.image, name);
    set_debug_name(alloc, VK_OBJECT_TYPE_IMAGE_VIEW, t.attachment_view, std::format("{}_attachment_view", name));
    vmaSetAllocationName(alloc, t.allocation, name.data());

    return t;
}


auto create_image_from_span_v2(VmaAllocator alloc, GlobalCommandContext &cmd_ctx, std::uint32_t width,
                               std::uint32_t height, VkFormat format, std::span<const std::byte> data,
                               std::string_view name) -> OffscreenTarget {
    std::span<const u8> data_as_u8 = std::span(std::bit_cast<const u8 *>(data.data()), data.size());
    return create_image_from_span_v2(alloc, cmd_ctx, width, height, format, data_as_u8, name);
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
        VkImageMemoryBarrier2 pre{};
        pre.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        pre.pNext = nullptr;
        pre.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        pre.srcAccessMask = 0;
        pre.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        pre.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        pre.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        pre.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        pre.image = t.image;
        pre.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .baseMipLevel = 0,
                                .levelCount = 1,
                                .baseArrayLayer = 0,
                                .layerCount = 1};

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

        VkImageMemoryBarrier2 post{};
        post.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        post.pNext = nullptr;
        post.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        post.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        post.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        post.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        post.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        post.newLayout = VK_IMAGE_LAYOUT_GENERAL; // We always use GENERAL. This is desktop safe.
        post.image = t.image;
        post.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                 .baseMipLevel = 0,
                                 .levelCount = 1,
                                 .baseArrayLayer = 0,
                                 .layerCount = 1};

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

auto create_depth_target(VmaAllocator &alloc, u32 width, u32 height, VkFormat format, VkSampleCountFlagBits samples,
                         bool want_sampled, // usually true only for single-sample depth you intend to sample later
                         std::string_view name) -> OffscreenTarget {
    OffscreenTarget t{};
    t.width = width;
    t.height = height;
    t.format = format;

    const VkImageUsageFlags usage = make_depth_image_usage(samples, want_sampled);

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {width, height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = samples;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(alloc, &ici, &aci, &t.image, &t.allocation, nullptr));

    VmaAllocatorInfo ai{};
    vmaGetAllocatorInfo(alloc, &ai);

    const VkImageAspectFlags aspect = choose_depth_aspect(format);

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = t.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                      VK_COMPONENT_SWIZZLE_IDENTITY};
    vci.subresourceRange = {
            .aspectMask = aspect, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1};

    // Attachment view always.
    vk_check(vkCreateImageView(ai.device, &vci, nullptr, &t.attachment_view));
    set_debug_name(alloc, VK_OBJECT_TYPE_IMAGE_VIEW, t.attachment_view, std::format("{}_attachment_view", name));

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

auto create_image_from_mips_v2(VmaAllocator alloc, GlobalCommandContext &cmd_ctx, u32 width, u32 height,
                               VkFormat format, std::span<const std::byte> data, std::span<const u32> mip_offsets,
                               std::span<const u32> mip_sizes, std::string_view name) -> OffscreenTarget {
    std::span<const u8> data_u8 = std::span(std::bit_cast<u8 const *>(data.data()), data.size());
    return create_image_from_mips_v2(alloc, cmd_ctx, width, height, format, data_u8, mip_offsets, mip_sizes, name);
}

auto is_block_compressed_format(VkFormat format) -> bool {
    switch (format) {
        case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
        case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
        case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
        case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
        case VK_FORMAT_BC2_UNORM_BLOCK:
        case VK_FORMAT_BC2_SRGB_BLOCK:
        case VK_FORMAT_BC3_UNORM_BLOCK:
        case VK_FORMAT_BC3_SRGB_BLOCK:
        case VK_FORMAT_BC4_UNORM_BLOCK:
        case VK_FORMAT_BC4_SNORM_BLOCK:
        case VK_FORMAT_BC5_UNORM_BLOCK:
        case VK_FORMAT_BC5_SNORM_BLOCK:
        case VK_FORMAT_BC6H_UFLOAT_BLOCK:
        case VK_FORMAT_BC6H_SFLOAT_BLOCK:
        case VK_FORMAT_BC7_UNORM_BLOCK:
        case VK_FORMAT_BC7_SRGB_BLOCK:
            return true;
        default:
            return false;
    }
}

auto create_texture_image_v2(VmaAllocator alloc, GlobalCommandContext &cmd_ctx, u32 width, u32 height, VkFormat format,
                             std::span<const u8> data, std::span<const u32> mip_offsets, std::span<const u32> mip_sizes,
                             std::string_view name) -> OffscreenTarget {
    u32 const mip_levels = static_cast<u32>(mip_offsets.size());
    if (mip_levels == 0 || mip_sizes.size() != mip_levels) {
        return create_image_from_mips_v2(alloc, cmd_ctx, width, height, format, data, mip_offsets, mip_sizes, name);
    }

    bool const is_compressed = is_block_compressed_format(format);

    if (is_compressed) {
        OffscreenTarget t{};
        t.width = width;
        t.height = height;
        t.format = format;

        VmaAllocatorInfo ai{};
        vmaGetAllocatorInfo(alloc, &ai);

        VkImageUsageFlags const usage =
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = format;
        ici.extent = {width, height, 1};
        ici.mipLevels = mip_levels;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(alloc, &ici, &aci, &t.image, &t.allocation, nullptr));

        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = t.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = format;
        vci.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                          VK_COMPONENT_SWIZZLE_IDENTITY};
        vci.subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = mip_levels,
                .baseArrayLayer = 0,
                .layerCount = 1,
        };

        vk_check(vkCreateImageView(ai.device, &vci, nullptr, &t.sampled_view));
        t.attachment_view = VK_NULL_HANDLE; // Not usable as attachment
        t.storage_view = VK_NULL_HANDLE; // BC7 doesn't support storage

        set_debug_name(alloc, VK_OBJECT_TYPE_IMAGE, t.image, name);
        set_debug_name(alloc, VK_OBJECT_TYPE_IMAGE_VIEW, t.sampled_view, std::format("{}_sampled_view", name));
        vmaSetAllocationName(alloc, t.allocation, name.data());

        // Now upload the data if provided
        if (!data.empty()) {
            // Validate bounds
            for (u32 i = 0; i < mip_levels; ++i) {
                u64 const end = u64(mip_offsets[i]) + u64(mip_sizes[i]);
                if (end > data.size_bytes()) {
                    return t; // bail out safely
                }
            }

            VmaAllocationCreateInfo staging_aci{};
            staging_aci.usage = VMA_MEMORY_USAGE_AUTO;
            staging_aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

            VkBufferCreateInfo bci{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .size = static_cast<VkDeviceSize>(data.size_bytes()),
                    .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                    .queueFamilyIndexCount = 0,
                    .pQueueFamilyIndices = nullptr,
            };

            VkBuffer staging{};
            VmaAllocation staging_alloc{};
            vk_check(vmaCreateBuffer(alloc, &bci, &staging_aci, &staging, &staging_alloc, nullptr));

            void *mapped{};
            vk_check(vmaMapMemory(alloc, staging_alloc, &mapped));
            std::memcpy(mapped, data.data(), data.size_bytes());
            vmaUnmapMemory(alloc, staging_alloc);

            auto submit_copy = [&](VkCommandBuffer cb) {
                // Transition to TRANSFER_DST
                auto pre = create_info<VkImageMemoryBarrier2>();
                pre.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
                pre.srcAccessMask = 0;
                pre.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                pre.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                pre.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                pre.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                pre.image = t.image;
                pre.subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = mip_levels,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                };

                auto di_pre = create_info<VkDependencyInfo>();
                di_pre.imageMemoryBarrierCount = 1;
                di_pre.pImageMemoryBarriers = &pre;

                vkCmdPipelineBarrier2(cb, &di_pre);

                // Copy each mip level
                std::vector<VkBufferImageCopy> copies;
                copies.reserve(mip_levels);

                for (u32 level = 0; level < mip_levels; ++level) {
                    u32 const lw = std::max(1u, width >> level);
                    u32 const lh = std::max(1u, height >> level);

                    VkBufferImageCopy bic{
                            .bufferOffset = static_cast<VkDeviceSize>(mip_offsets[level]),
                            .bufferRowLength = 0,
                            .bufferImageHeight = 0,
                            .imageSubresource =
                                    {
                                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                            .mipLevel = level,
                                            .baseArrayLayer = 0,
                                            .layerCount = 1,
                                    },
                            .imageOffset = {0, 0, 0},
                            .imageExtent = {lw, lh, 1},
                    };

                    copies.push_back(bic);
                }

                vkCmdCopyBufferToImage(cb, staging, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                       static_cast<u32>(copies.size()), copies.data());

                // Transition to GENERAL for sampling
                auto post = create_info<VkImageMemoryBarrier2>();
                post.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                post.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                post.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                post.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                post.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                post.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                post.image = t.image;
                post.subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = mip_levels,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                };


                auto di_post = create_info<VkDependencyInfo>();
                di_post.imageMemoryBarrierCount = 1;
                di_post.pImageMemoryBarriers = &post;

                vkCmdPipelineBarrier2(cb, &di_post);
            };

            submit_one_time_cmd(cmd_ctx, submit_copy, true);
            t.initialized = true;

            vmaDestroyBuffer(alloc, staging, staging_alloc);
        }

        return t;
    } else {
        // Uncompressed format: use the standard path
        return create_image_from_mips_v2(alloc, cmd_ctx, width, height, format, data, mip_offsets, mip_sizes, name);
    }
}

auto create_image_from_mips_v2(VmaAllocator alloc, GlobalCommandContext &cmd_ctx, u32 width, u32 height,
                               VkFormat format, std::span<const u8> data, std::span<const u32> mip_offsets,
                               std::span<const u32> mip_sizes, std::string_view name) -> OffscreenTarget {
    u32 const mip_levels = static_cast<u32>(mip_offsets.size());
    if (mip_levels == 0 || mip_sizes.size() != mip_levels) {
        // fallback: create empty
        return create_offscreen_target(alloc, width, height, format, VK_SAMPLE_COUNT_1_BIT, {}, name);
    }

    // You need create_offscreen_target to accept mip_levels (or provide a new helper).
    // I’m assuming you can add a mip_levels parameter; otherwise you must modify it internally.
    auto t = create_offscreen_target(alloc, width, height, format, VK_SAMPLE_COUNT_1_BIT,
                                     TargetSamplerConfiguration{.dims = {.mip_levels = mip_levels}}, name);

    if (data.empty()) {
        return t;
    }

    // Validate bounds (cheap sanity)
    for (u32 i = 0; i < mip_levels; ++i) {
        u64 const end = u64(mip_offsets[i]) + u64(mip_sizes[i]);
        if (end > data.size_bytes()) {
            // bail out safely: return empty texture
            return t;
        }
    }

    VmaAllocationCreateInfo staging_aci{};
    staging_aci.usage = VMA_MEMORY_USAGE_AUTO;
    staging_aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    VkBufferCreateInfo bci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = static_cast<VkDeviceSize>(data.size_bytes()),
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
    };

    VkBuffer staging{};
    VmaAllocation staging_alloc{};
    vk_check(vmaCreateBuffer(alloc, &bci, &staging_aci, &staging, &staging_alloc, nullptr));

    void *mapped{};
    vk_check(vmaMapMemory(alloc, staging_alloc, &mapped));
    std::memcpy(mapped, data.data(), data.size_bytes());
    vmaUnmapMemory(alloc, staging_alloc);

    auto submit_copy = [&](VkCommandBuffer cb) {
        // Transition all mip levels to TRANSFER_DST_OPTIMAL
        VkImageMemoryBarrier2 pre{};
        pre.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        pre.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        pre.srcAccessMask = 0;
        pre.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        pre.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        pre.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        pre.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        pre.image = t.image;
        pre.subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = mip_levels,
                .baseArrayLayer = 0,
                .layerCount = 1,
        };


        auto di_pre = create_info<VkDependencyInfo>();
        di_pre.imageMemoryBarrierCount = 1;
        di_pre.pImageMemoryBarriers = &pre;

        vkCmdPipelineBarrier2(cb, &di_pre);

        // Copy each mip level
        std::vector<VkBufferImageCopy> copies;
        copies.reserve(mip_levels);

        for (u32 level = 0; level < mip_levels; ++level) {
            VkExtent3D const ext = mip_extent(width, height, level);

            VkBufferImageCopy bic{
                    .bufferOffset = static_cast<VkDeviceSize>(mip_offsets[level]),
                    .bufferRowLength = 0,
                    .bufferImageHeight = 0,
                    .imageSubresource =
                            {
                                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                    .mipLevel = level,
                                    .baseArrayLayer = 0,
                                    .layerCount = 1,
                            },
                    .imageOffset = {0, 0, 0},
                    .imageExtent = ext,
            };

            copies.push_back(bic);
        }

        vkCmdCopyBufferToImage(cb, staging, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               static_cast<u32>(copies.size()), copies.data());

        // Transition all mips to GENERAL for sampling
        VkImageMemoryBarrier2 post{};
        post.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        post.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        post.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        post.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        post.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        post.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        post.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        post.image = t.image;
        post.subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = mip_levels,
                .baseArrayLayer = 0,
                .layerCount = 1,
        };

        auto di_post = create_info<VkDependencyInfo>();
        di_post.imageMemoryBarrierCount = 1;
        di_post.pImageMemoryBarriers = &post;

        vkCmdPipelineBarrier2(cb, &di_post);
    };

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

        std::optional<u32> transfer_dedicated{};
        std::optional<u32> transfer_shared_no_graphics{};
        std::optional<u32> transfer_shared_any{};

        for (u32 i = 0u; i < qcount; ++i) {
            VkQueueFlags flags = qprops[i].queueFlags;

            // Track transfer candidates
            if (flags & VK_QUEUE_TRANSFER_BIT) {
                // Pure transfer (best for async copies)
                if (!(flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) {
                    if (!transfer_dedicated) {
                        transfer_dedicated = i;
                    }
                } else if (!(flags & VK_QUEUE_GRAPHICS_BIT)) {
                    // Transfer on a non-graphics queue (often compute queue) is usually nicer than graphics
                    if (!transfer_shared_no_graphics) {
                        transfer_shared_no_graphics = i;
                    }
                }

                if (!transfer_shared_any) {
                    transfer_shared_any = i;
                }
            }

            // Graphics + potential shared compute
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

            // Dedicated compute (non-graphics)
            if (flags & VK_QUEUE_COMPUTE_BIT) {
                if (!(flags & VK_QUEUE_GRAPHICS_BIT)) {
                    if (!compute_dedicated) {
                        compute_dedicated = i;
                    }
                }
            }
        }

        auto pick_transfer_family = [&]() -> std::optional<u32> {
            if (transfer_dedicated) {
                return transfer_dedicated;
            }
            if (transfer_shared_no_graphics) {
                return transfer_shared_no_graphics;
            }
            return transfer_shared_any;
        };

        std::optional<u32> transfer = pick_transfer_family();
        if (!transfer) {
            // TODO: Is this really the best choice?
            transfer = graphics;
            continue;
        }

        if (graphics && compute_dedicated) {
            return DeviceChoice{pd, *graphics, *compute_dedicated, *transfer};
        }

        if (graphics && compute_shared) {
            return DeviceChoice{pd, *graphics, *compute_shared, *transfer};
        }
    }

    return tl::unexpected(PhysicalDeviceChoice{PhysicalDeviceChoice::Error::NoQueuesFound});
}


namespace {
    auto collect_supported_extensions(VkPhysicalDevice pd) -> std::unordered_set<std::string> {
        u32 ext_count{};
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &ext_count, nullptr);

        std::vector<VkExtensionProperties> props(ext_count);
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &ext_count, props.data());

        std::unordered_set<std::string> out;
        out.reserve(ext_count);

        for (auto const &e: props) {
            out.insert(e.extensionName);
        }
        return out;
    }

    auto enable_extensions(std::unordered_set<std::string> const &supported, std::span<char const *const> desired,
                           std::vector<char const *> &enabled_exts, EnabledFeatureSet &enabled_set) -> void {
        for (auto const *name: desired) {
            if (supported.contains(name)) {
                enabled_exts.push_back(name);
                enabled_set.insert(name);
                info("Enabling device extension '{}'.", name);
            } else {
                info("Device extension '{}' not supported; skipping.", name);
            }
        }
    }
} // namespace

auto create_device(VkPhysicalDevice pd, u32 graphics_index, u32 compute_index, u32 transfer_index)
        -> std::tuple<VkDevice, VkQueue, VkQueue, VkQueue, EnabledFeatureSet> {
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

    auto add_if_supported = [&](char const *name, std::vector<char const *> &out, EnabledFeatureSet &features) -> void {
        if (auto str = std::string(name); has_ext(str.c_str())) {
            out.push_back(name);
            features.insert(std::move(str));
            info("Enabling device extension '{}'.", name);
        } else {
            info("Device extension '{}' not supported; skipping.", name);
        }
    };

    auto supported = collect_supported_extensions(pd);

    EnabledFeatureSet enabled_features;
    std::vector<char const *> enabled_exts;

    constexpr std::array desired_exts{
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            VK_EXT_MESH_SHADER_EXTENSION_NAME,
            VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME,
            VK_KHR_MAINTENANCE_9_EXTENSION_NAME,
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    };


    enable_extensions(supported, desired_exts, enabled_exts, enabled_features);

    add_if_supported(VK_NV_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME, enabled_exts, enabled_features);

    VkPhysicalDeviceFragmentShadingRateFeaturesKHR shading_rate_features_khr{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR,
            .pNext = nullptr,
            .pipelineFragmentShadingRate = VK_TRUE,
            .primitiveFragmentShadingRate = VK_TRUE,
            .attachmentFragmentShadingRate = VK_TRUE};

    VkPhysicalDeviceMeshShaderFeaturesEXT mesh_features{};
    mesh_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
    mesh_features.pNext = &shading_rate_features_khr;
    mesh_features.taskShader = VK_TRUE; // Optional; but recommended for culling
    mesh_features.meshShader = VK_TRUE;
    mesh_features.primitiveFragmentShadingRateMeshShader = VK_TRUE;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_features{};
    accel_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accel_features.pNext = &mesh_features;

    VkPhysicalDeviceVulkan11Features features11{};
    features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, features11.pNext = &accel_features;

    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, features12.pNext = &features11;

    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, features13.pNext = &features12;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, features2.pNext = &features13;


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

    if (enabled_features.contains(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
        enabled_features.contains(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)) {
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
    float priority_transfer = 1.0f;
    VkDeviceQueueCreateInfo qci_transfer{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                         .pNext = nullptr,
                                         .flags = 0,
                                         .queueFamilyIndex = transfer_index,
                                         .queueCount = 1u,
                                         .pQueuePriorities = &priority_transfer};

    std::array<VkDeviceQueueCreateInfo, 3> qcis{qci_graphics, qci_compute, qci_transfer};

    u32 qci_count = 0u;
    qcis[qci_count++] = qci_graphics;
    if (compute_index != graphics_index) {
        qcis[qci_count++] = qci_compute;
    }
    if (transfer_index != graphics_index && transfer_index != compute_index) {
        qcis[qci_count++] = qci_transfer;
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

    VkQueue tq{};
    vkGetDeviceQueue(device, transfer_index, 0u, &tq);

    return {device, gq, cq, tq, enabled_features};
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

auto throttle(auto &tl, VkDevice device) -> void {
    u64 current = 0;
    vk_check(vkGetSemaphoreCounterValue(device, tl.timeline, &current));
    tl.completed = current;

    const u64 limit = max_in_flight_submits<std::decay_t<decltype(tl)>>();
    if (tl.value <= tl.completed + limit)
        return;

    const u64 wait_val = tl.value - limit;
    const VkSemaphoreWaitInfo wi{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                 .pNext = nullptr,
                                 .flags = 0,
                                 .semaphoreCount = 1,
                                 .pSemaphores = &tl.timeline,
                                 .pValues = &wait_val};
    vk_check(vkWaitSemaphores(device, &wi, UINT64_MAX));
    tl.completed = wait_val;
}

auto throttle(ComputeTimeline &tl, VkDevice device) -> void { return throttle<ComputeTimeline>(tl, device); }

auto throttle(GraphicsTimeline &tl, VkDevice device) -> void { return throttle<GraphicsTimeline>(tl, device); }
