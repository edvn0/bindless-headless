#include "ImageOperations.hxx"
#include "Logger.hxx"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <glm/gtc/packing.hpp>
#include <vector>


#include "ImageWriter.hxx" // declares CpuImage, PixelLayout, IImageWriter, make_image_writer_from_filename
#include "Profiler.hxx"
#include "Types.hxx"


#ifdef __AVX2__
#include <immintrin.h>
#endif


namespace {

    auto channel_count(PixelLayout layout) -> u32 { return layout == PixelLayout::Rgba8 ? 4u : 3u; }

} // namespace

namespace {

    auto bytes_per_pixel(VkFormat format) -> u32 {
        switch (format) {
            // RGBA8
            case VK_FORMAT_R8G8B8A8_UNORM:
            case VK_FORMAT_R8G8B8A8_SRGB:
                return 4;

                // RGBA16 (4 * 16-bit = 8 bytes)
            case VK_FORMAT_R16G16B16A16_UNORM:
            case VK_FORMAT_R16G16B16A16_SNORM:
            case VK_FORMAT_R16G16B16A16_UINT:
            case VK_FORMAT_R16G16B16A16_SINT:
            case VK_FORMAT_R16G16B16A16_SFLOAT:
                return 8;

                // RGBA32 (4 * 32-bit = 16 bytes)
            case VK_FORMAT_R32G32B32A32_SFLOAT:
            case VK_FORMAT_R32G32B32A32_UINT:
            case VK_FORMAT_R32G32B32A32_SINT:
                return 16;

                // R8
            case VK_FORMAT_R8_UNORM:
            case VK_FORMAT_R8_SRGB:
            case VK_FORMAT_R8_UINT:
            case VK_FORMAT_R8_SINT:
                return 1;

            default:
                return 0;
        }
    }

    auto calc_tightly_packed_image_size_bytes(u32 width, u32 height, VkFormat format) -> VkDeviceSize {
        u32 const bpp = bytes_per_pixel(format);
        return VkDeviceSize(width) * VkDeviceSize(height) * VkDeviceSize(bpp);
    }

} // namespace


namespace image_operations {

    namespace {
        constexpr u32 SRGB_LUT_SIZE = 4096;
        std::array<u8, SRGB_LUT_SIZE> g_srgb_lut{};
        std::once_flag g_srgb_lut_once;

        void init_srgb_lut() {
            std::call_once(g_srgb_lut_once, [] {
                ZoneScopedNC("init_srgb_lut", 0xAAAAFF);

                for (u32 i = 0; i < SRGB_LUT_SIZE; ++i) {
                    float v = float(i) / float(SRGB_LUT_SIZE - 1);

                    if (v <= 0.0031308f)
                        v = 12.92f * v;
                    else
                        v = 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;

                    v = std::clamp(v, 0.0f, 1.0f);
                    g_srgb_lut[i] = u8(v * 255.0f + 0.5f);
                }
            });
        }

        u8 float_to_srgb(float v) {
            if (!std::isfinite(v))
                return 0;

            v = v / (1.0f + v);
            v = std::clamp(v, 0.0f, 1.0f);

            u32 idx = u32(v * (SRGB_LUT_SIZE - 1));
            return g_srgb_lut[idx];
        }

        // ============================
        // AVX2 conversion helpers
        // ============================

#ifdef __AVX2__
        void convert_rgba8_row_avx2(u8 *dst, const u8 *src, u32 width) {
            u32 x = 0;

            // Process 8 pixels at a time with AVX2
            for (; x + 8 <= width; x += 8) {
                // Load 8 RGBA pixels (32 bytes)
                __m256i rgba = _mm256_loadu_si256((__m256i *) (src + x * 4));

                // Shuffle to BGR layout
                const __m256i shuffle_mask = _mm256_setr_epi8(2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1, 2,
                                                              1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1);

                __m256i bgr_shuffled = _mm256_shuffle_epi8(rgba, shuffle_mask);

                // Extract and store (fallback to scalar for now)
                u8 temp[32];
                _mm256_storeu_si256((__m256i *) temp, bgr_shuffled);

                for (u32 i = 0; i < 8; ++i) {
                    dst[i * 3 + 0] = src[x * 4 + i * 4 + 2];
                    dst[i * 3 + 1] = src[x * 4 + i * 4 + 1];
                    dst[i * 3 + 2] = src[x * 4 + i * 4 + 0];
                }
                dst += 24;
            }

            // Handle remaining pixels
            for (; x < width; ++x) {
                dst[0] = src[2];
                dst[1] = src[1];
                dst[2] = src[0];
                src += 4;
                dst += 3;
            }
        }
#endif

        // ============================
        // Format-specific converters (OpenMP versions)
        // ============================

        auto half_to_float(u16 h) -> float { return glm::unpackHalf1x16(h); }

    } // anonymous namespace

    namespace {

        void convert_rgba8(CpuImage &out, u8 const *src, u32 width, u32 height, PixelLayout layout) {
            u32 const dst_channels = channel_count(layout);

            out.width = width;
            out.height = height;
            out.layout = layout;
            out.stride_bytes = width * dst_channels;
            out.pixels.resize(std::size_t(out.stride_bytes) * height);

            u32 const src_stride = width * 4;

#pragma omp parallel for schedule(dynamic, 16)
            for (i32 y = 0; y < static_cast<i32>(height); ++y) {
                u8 const *s = src + std::size_t(y) * src_stride; // top-down
                u8 *d = out.pixels.data() + std::size_t(y) * out.stride_bytes;

                for (u32 x = 0; x < width; ++x) {
                    // src RGBA
                    d[0] = s[0];
                    d[1] = s[1];
                    d[2] = s[2];
                    if (dst_channels == 4)
                        d[3] = s[3];

                    s += 4;
                    d += dst_channels;
                }
            }
        }

        void convert_rgba32f(CpuImage &out, float const *src, u32 width, u32 height, PixelLayout layout) {
            init_srgb_lut();

            u32 const dst_channels = channel_count(layout);

            out.width = width;
            out.height = height;
            out.layout = layout;
            out.stride_bytes = width * dst_channels;
            out.pixels.resize(std::size_t(out.stride_bytes) * height);

#pragma omp parallel for schedule(dynamic, 16)
            for (i32 y = 0; y < static_cast<i32>(height); ++y) {
                float const *s = src + std::size_t(y) * width * 4; // top-down
                u8 *d = out.pixels.data() + std::size_t(y) * out.stride_bytes;

                for (u32 x = 0; x < width; ++x) {
                    d[0] = float_to_srgb(s[0]);
                    d[1] = float_to_srgb(s[1]);
                    d[2] = float_to_srgb(s[2]);
                    if (dst_channels == 4)
                        d[3] = 255;

                    s += 4;
                    d += dst_channels;
                }
            }
        }

        void convert_r8(CpuImage &out, u8 const *src, u32 width, u32 height, PixelLayout layout) {
            u32 const dst_channels = channel_count(layout);

            out.width = width;
            out.height = height;
            out.layout = layout;
            out.stride_bytes = width * dst_channels;
            out.pixels.resize(std::size_t(out.stride_bytes) * height);

            u32 const src_stride = width;

#pragma omp parallel for schedule(dynamic, 16)
            for (i32 y = 0; y < static_cast<i32>(height); ++y) {
                u8 const *s = src + std::size_t(y) * src_stride;
                u8 *d = out.pixels.data() + std::size_t(y) * out.stride_bytes;

                for (u32 x = 0; x < width; ++x) {
                    u8 v = s[x];
                    d[0] = v;
                    d[1] = v;
                    d[2] = v;
                    if (dst_channels == 4)
                        d[3] = 255;
                    d += dst_channels;
                }
            }
        }

        void convert_rgba16f(CpuImage &out, u8 const *src, u32 width, u32 height, PixelLayout layout) {
            init_srgb_lut();

            u32 const dst_channels = channel_count(layout);

            out.width = width;
            out.height = height;
            out.layout = layout;
            out.stride_bytes = width * dst_channels;
            out.pixels.resize(std::size_t(out.stride_bytes) * height);

#pragma omp parallel for schedule(dynamic, 16)
            for (i32 y = 0; y < static_cast<i32>(height); ++y) {
                auto const *s = reinterpret_cast<u16 const *>(src + std::size_t(y) * width * 8);
                u8 *d = out.pixels.data() + std::size_t(y) * out.stride_bytes;

                for (u32 x = 0; x < width; ++x) {
                    float r = half_to_float(s[0]);
                    float g = half_to_float(s[1]);
                    float b = half_to_float(s[2]);

                    d[0] = float_to_srgb(r);
                    d[1] = float_to_srgb(g);
                    d[2] = float_to_srgb(b);
                    if (dst_channels == 4)
                        d[3] = 255;

                    s += 4;
                    d += dst_channels;
                }
            }
        }

        auto convert_pixels(CpuImage &out, u8 const *pixel_data, u32 width, u32 height, VkFormat format,
                            PixelLayout layout) -> bool {
            switch (format) {
                case VK_FORMAT_R8G8B8A8_UNORM:
                case VK_FORMAT_R8G8B8A8_SRGB:
                    convert_rgba8(out, pixel_data, width, height, layout);
                    return true;

                case VK_FORMAT_R32G32B32A32_SFLOAT:
                    convert_rgba32f(out, reinterpret_cast<float const *>(pixel_data), width, height, layout);
                    return true;

                case VK_FORMAT_R8_UNORM:
                case VK_FORMAT_R8_SRGB:
                case VK_FORMAT_R8_UINT:
                case VK_FORMAT_R8_SINT:
                    convert_r8(out, pixel_data, width, height, layout);
                    return true;

                case VK_FORMAT_R16G16B16A16_SFLOAT:
                case VK_FORMAT_R16G16B16A16_UNORM:
                case VK_FORMAT_R16G16B16A16_SNORM:
                case VK_FORMAT_R16G16B16A16_UINT:
                case VK_FORMAT_R16G16B16A16_SINT:
                    convert_rgba16f(out, pixel_data, width, height, layout);
                    return true;

                default:
                    return false;
            }
        }
    } // namespace

    // ============================
    // Public entry
    // ============================

    struct StagingBuffer {
        VkBuffer buffer;
        VmaAllocation allocation;
        VmaAllocationInfo alloc_info;
        u32 width;
        u32 height;
        VkFormat format;
        u32 request_index;
    };

    // Batch write multiple images (OpenMP version)
    void write_batch_to_disk(VmaAllocator &allocator, std::span<const ImageWriteRequest> requests,
                             ProgressFn report_progress) {
        ZoneScopedNC("write_batch_to_disk", 0x8050FF);

        if (requests.empty()) {
            return;
        }

        VmaAllocatorInfo allocator_info{};
        vmaGetAllocatorInfo(allocator, &allocator_info);

        std::vector<StagingBuffer> staging_buffers;
        staging_buffers.reserve(requests.size());

        {
            ZoneScopedNC("create_staging_buffers", 0x4080FF);

            for (u32 req_index = 0; req_index < requests.size(); ++req_index) {
                auto const &req = requests[req_index];
                if (!req.texture) {
                    error("Null texture for {}", req.filename);
                    continue;
                }

                auto const &tex = *req.texture;

                if (auto pixel_size = bytes_per_pixel(tex.format); pixel_size == 0) {
                    error("Unsupported format for {}: {}", req.filename, static_cast<u32>(tex.format));
                    continue;
                }
                const auto buffer_size = calc_tightly_packed_image_size_bytes(tex.width, tex.height, tex.format);

                VkBufferCreateInfo buffer_create_info{
                        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                        .pNext = nullptr,
                        .flags = 0,
                        .size = buffer_size,
                        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                        .queueFamilyIndexCount = 0,
                        .pQueueFamilyIndices = nullptr,
                };

                VmaAllocationCreateInfo alloc_create_info{};
                alloc_create_info.flags =
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
                alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;

                StagingBuffer staging{};
                auto result = vmaCreateBuffer(allocator, &buffer_create_info, &alloc_create_info, &staging.buffer,
                                              &staging.allocation, &staging.alloc_info);

                if (result != VK_SUCCESS) {
                    error("Failed to create staging buffer for {}: {}", req.filename, static_cast<u32>(result));
                    continue;
                }

                staging.width = tex.width;
                staging.height = tex.height;
                staging.format = tex.format;
                staging.request_index = req_index;
                staging_buffers.push_back(staging);
            }
        }

        if (staging_buffers.empty()) {
            return;
        }

        VkCommandPool command_pool{};
        {
            ZoneScopedNC("create_command_pool", 0x4080FF);

            VkCommandPoolCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            info.queueFamilyIndex = 0;

            if (vkCreateCommandPool(allocator_info.device, &info, nullptr, &command_pool) != VK_SUCCESS) {
                error("Failed to create command pool");
                for (auto const &sb: staging_buffers) {
                    vmaDestroyBuffer(allocator, sb.buffer, sb.allocation);
                }
                return;
            }
        }

        VkCommandBuffer command_buffer{};
        {
            ZoneScopedNC("allocate_command_buffer", 0x4080FF);

            VkCommandBufferAllocateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            info.commandPool = command_pool;
            info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            info.commandBufferCount = 1;

            if (vkAllocateCommandBuffers(allocator_info.device, &info, &command_buffer) != VK_SUCCESS) {
                error("Failed to allocate command buffer");
                vkDestroyCommandPool(allocator_info.device, command_pool, nullptr);
                for (auto const &sb: staging_buffers) {
                    vmaDestroyBuffer(allocator, sb.buffer, sb.allocation);
                }
                return;
            }
        }

        {
            ZoneScopedNC("record_batch_commands", 0x40FFFF);

            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            vkBeginCommandBuffer(command_buffer, &begin_info);

            for (auto const &staging: staging_buffers) {
                auto const &req = requests[staging.request_index];
                auto const &tex = *req.texture;

                VkImageMemoryBarrier2 barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                barrier.srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                barrier.image = tex.image;
                barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

                VkDependencyInfo dep{};
                dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.imageMemoryBarrierCount = 1;
                dep.pImageMemoryBarriers = &barrier;

                vkCmdPipelineBarrier2(command_buffer, &dep);

                // Copy image to buffer
                VkBufferImageCopy region{};
                region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.imageExtent = {tex.width, tex.height, 1};

                vkCmdCopyImageToBuffer(command_buffer, tex.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging.buffer,
                                       1, &region);

                // Transition back to general
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
                barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;

                vkCmdPipelineBarrier2(command_buffer, &dep);
            }

            vkEndCommandBuffer(command_buffer);
        }

        // Submit and wait
        {
            ZoneScopedNC("submit_batch_and_wait", 0xFFAA40);

            VkFence fence{};
            VkFenceCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            vkCreateFence(allocator_info.device, &info, nullptr, &fence);

            VkQueue queue{};
            vkGetDeviceQueue(allocator_info.device, 0, 0, &queue);

            VkSubmitInfo submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command_buffer;

            vkQueueSubmit(queue, 1, &submit, fence);
            vkWaitForFences(allocator_info.device, 1, &fence, VK_TRUE, UINT64_MAX);

            vkDestroyFence(allocator_info.device, fence, nullptr);
        }

        // Now process all images in parallel on CPU using OpenMP
        {
            ZoneScopedNC("parallel_cpu_processing", 0x40FF40);

            std::atomic<u32> images_done = 0;
            const u32 total_images = static_cast<u32>(staging_buffers.size());

            for (auto &sb: staging_buffers) {
                // Ensure memory is visible
                vmaFlushAllocation(allocator, sb.allocation, 0, VK_WHOLE_SIZE);
            }

#pragma omp parallel for schedule(dynamic)
            for (auto i = 0; i < static_cast<i32>(staging_buffers.size()); ++i) {
                ZoneScopedNC("process_single_image", 0xFF40FF);

                auto const &staging = staging_buffers[i];
                auto const &req = requests[staging.request_index];


                auto writer = make_image_writer_from_filename(req.filename);
                if (!writer) {
                    error("No writer for {}", req.filename);
                    continue;
                }

                // Decide desired pixel layout based on writer / extension.
                // Minimal: png => RGBA, else RGB.
                PixelLayout layout = PixelLayout::Rgb8;
                if (writer->extension() == "png")
                    layout = PixelLayout::Rgba8;

                auto pixel_data = static_cast<u8 *>(staging.alloc_info.pMappedData);

                CpuImage img;
                if (!convert_pixels(img, pixel_data, staging.width, staging.height, staging.format, layout)) {
                    error("Unsupported format for {}: {}", req.filename, static_cast<u32>(staging.format));
                    continue;
                }

                if (!writer->write(req.filename, img)) {
                    error("Failed to write {}", req.filename);
                    continue;
                }

                auto done = ++images_done;

                // Optional callback / logging
                if (report_progress)
                    report_progress(float(done) / float(total_images));
            }
        }

        {
            ZoneScopedNC("batch_cleanup", 0x808080);
            vkDestroyCommandPool(allocator_info.device, command_pool, nullptr);
            for (auto const &sb: staging_buffers) {
                vmaDestroyBuffer(allocator, sb.buffer, sb.allocation);
            }
        }
    }

    void write_to_disk(const OffscreenTarget *texture, VmaAllocator &allocator, std::string_view filename) {
        ZoneScopedNC("write_to_disk", 0x8050FF);

        if (!texture) {
            error("Null texture");
            return;
        }


        const auto &tex = *texture;

        const u32 row_size = ((tex.width * 3 + 3) / 4) * 4;
        std::vector<u8> cpu_pixels(row_size * tex.height);

        if (auto pixel_size = bytes_per_pixel(tex.format); pixel_size == 0) {
            error("Unsupported format for writing to disk: {}", static_cast<u32>(tex.format));
            return;
        }
        const auto buffer_size = calc_tightly_packed_image_size_bytes(tex.width, tex.height, tex.format);

        VmaAllocatorInfo allocator_info{};
        vmaGetAllocatorInfo(allocator, &allocator_info);


        VkBuffer staging_buffer{};
        VmaAllocation staging_allocation{};
        VmaAllocationInfo staging_alloc_info{};

        {
            ZoneScopedNC("create_staging_buffer", 0x4080FF);

            VkBufferCreateInfo buffer_create_info{};
            buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buffer_create_info.size = buffer_size;
            buffer_create_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo alloc_create_info{};
            alloc_create_info.flags =
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;

            auto result = vmaCreateBuffer(allocator, &buffer_create_info, &alloc_create_info, &staging_buffer,
                                          &staging_allocation, &staging_alloc_info);

            if (result != VK_SUCCESS) {
                error("Failed to create staging buffer: {}", static_cast<u32>(result));
                return;
            }
        }

        VkCommandPool command_pool{};
        {
            ZoneScopedNC("create_command_pool", 0x4080FF);

            VkCommandPoolCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            info.queueFamilyIndex = 0;

            if (vkCreateCommandPool(allocator_info.device, &info, nullptr, &command_pool) != VK_SUCCESS) {
                vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
                return;
            }
        }

        VkCommandBuffer command_buffer{};
        {
            ZoneScopedNC("allocate_command_buffer", 0x4080FF);

            VkCommandBufferAllocateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            info.commandPool = command_pool;
            info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            info.commandBufferCount = 1;

            if (vkAllocateCommandBuffers(allocator_info.device, &info, &command_buffer) != VK_SUCCESS) {
                vkDestroyCommandPool(allocator_info.device, command_pool, nullptr);
                vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
                return;
            }
        }

        {
            ZoneScopedNC("record_commands", 0x40FFFF);

            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            vkBeginCommandBuffer(command_buffer, &begin_info);

            VkImageMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.image = tex.image;
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &barrier;

            vkCmdPipelineBarrier2(command_buffer, &dep);

            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {tex.width, tex.height, 1};

            vkCmdCopyImageToBuffer(command_buffer, tex.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging_buffer, 1,
                                   &region);

            barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;

            vkCmdPipelineBarrier2(command_buffer, &dep);
            vkEndCommandBuffer(command_buffer);
        }

        {
            ZoneScopedNC("submit_and_wait", 0xFFAA40);

            VkFence fence{};
            VkFenceCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            vkCreateFence(allocator_info.device, &info, nullptr, &fence);

            VkQueue queue{};
            vkGetDeviceQueue(allocator_info.device, 0, 0, &queue);

            VkSubmitInfo submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command_buffer;

            vkQueueSubmit(queue, 1, &submit, fence);
            vkWaitForFences(allocator_info.device, 1, &fence, VK_TRUE, UINT64_MAX);

            vkDestroyFence(allocator_info.device, fence, nullptr);
        }

        vmaInvalidateAllocation(allocator, staging_allocation, 0, VK_WHOLE_SIZE);

        const u8 *pixel_data = static_cast<const u8 *>(staging_alloc_info.pMappedData);

        auto writer = make_image_writer_from_filename(filename);
        if (!writer) {
            error("No writer for {}", filename);
            return;
        }

        PixelLayout layout = PixelLayout::Rgb8;
        if (writer->extension() == "png")
            layout = PixelLayout::Rgba8;

        CpuImage img;
        if (!convert_pixels(img, pixel_data, tex.width, tex.height, tex.format, layout)) {
            error("Unsupported format for writing to disk: {}", static_cast<u32>(tex.format));
            return;
        }

        if (!writer->write(filename, img)) {
            error("Failed to write {}", filename);
            return;
        }

        {
            ZoneScopedNC("cleanup", 0x808080);
            vkDestroyCommandPool(allocator_info.device, command_pool, nullptr);
            vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
        }
    }

} // namespace image_operations
