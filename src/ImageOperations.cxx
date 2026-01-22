#include "ImageOperations.hxx"
#include "Logger.hxx"

#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <vector>

#include <tracy/Tracy.hpp>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace image_operations {

    namespace {
        constexpr u32 SRGB_LUT_SIZE = 4096;
        static std::array<u8, SRGB_LUT_SIZE> g_srgb_lut{};
        static std::once_flag g_srgb_lut_once;

        static void init_srgb_lut() {
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

        inline u8 float_to_srgb(float v) {
            if (!std::isfinite(v))
                return 0;

            v = v / (1.0f + v);
            v = std::clamp(v, 0.0f, 1.0f);

            u32 idx = u32(v * (SRGB_LUT_SIZE - 1));
            return g_srgb_lut[idx];
        }

        // ============================
        // BMP header
        // ============================

        void write_bmp_headers(std::ofstream &output, u32 width, u32 height) {
            ZoneScopedNC("write_bmp_headers", 0x8080FF);

            const u32 row_size = ((width * 3 + 3) / 4) * 4;
            const u32 pixel_bytes = row_size * height;
            const u32 file_size = 14 + 40 + pixel_bytes;

            const u16 bf_type = 0x4D42;
            const u32 bf_off_bits = 14 + 40;

            output.write((char *) &bf_type, 2);
            output.write((char *) &file_size, 4);

            u32 reserved = 0;
            output.write((char *) &reserved, 4);
            output.write((char *) &bf_off_bits, 4);

            u32 bi_size = 40;
            i32 bi_width = i32(width);
            i32 bi_height = i32(height);
            u16 bi_planes = 1;
            u16 bi_bit_count = 24;
            u32 bi_compression = 0;
            u32 bi_size_image = pixel_bytes;
            i32 ppm = 0;
            u32 clr = 0;

            output.write((char *) &bi_size, 4);
            output.write((char *) &bi_width, 4);
            output.write((char *) &bi_height, 4);
            output.write((char *) &bi_planes, 2);
            output.write((char *) &bi_bit_count, 2);
            output.write((char *) &bi_compression, 4);
            output.write((char *) &bi_size_image, 4);
            output.write((char *) &ppm, 4);
            output.write((char *) &ppm, 4);
            output.write((char *) &clr, 4);
            output.write((char *) &clr, 4);
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
                // We need to extract R, G, B from RGBA and pack them
                const __m256i shuffle_mask = _mm256_setr_epi8(2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1, 2,
                                                              1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1);

                __m256i bgr_shuffled = _mm256_shuffle_epi8(rgba, shuffle_mask);

                // Extract and store (this is complex, fallback to scalar for now)
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
        // Format-specific converters
        // ============================

        void convert_rgba8_rows(u8 *dst, const u8 *src, u32 width, u32 height, u32 y_begin, u32 y_end) {
            ZoneScopedNC("convert_rgba8_rows", 0x40FF80);

            const u32 dst_stride = ((width * 3 + 3) / 4) * 4;
            const u32 src_stride = width * 4;

            for (u32 y = y_begin; y < y_end; ++y) {
                const u8 *s = src + (height - 1 - y) * src_stride;
                u8 *d = dst + y * dst_stride;

#ifdef __AVX2__
                convert_rgba8_row_avx2(d, s, width);
#else
                for (u32 x = 0; x < width; ++x) {
                    d[0] = s[2];
                    d[1] = s[1];
                    d[2] = s[0];
                    s += 4;
                    d += 3;
                }
#endif
            }
        }

        void convert_rgba32f_rows(u8 *dst, const float *src, u32 width, u32 height, u32 y_begin, u32 y_end) {
            ZoneScopedNC("convert_rgba32f_rows", 0xFF8844);

            const u32 dst_stride = ((width * 3 + 3) / 4) * 4;

            for (u32 y = y_begin; y < y_end; ++y) {
                const float *s = src + (height - 1 - y) * width * 4;
                u8 *d = dst + y * dst_stride;

                for (u32 x = 0; x < width; ++x) {
                    d[0] = float_to_srgb(s[2]);
                    d[1] = float_to_srgb(s[1]);
                    d[2] = float_to_srgb(s[0]);
                    s += 4;
                    d += 3;
                }
            }
        }

        void convert_r8_rows(u8 *dst, const u8 *src, u32 width, u32 height, u32 y_begin, u32 y_end) {
            ZoneScopedNC("convert_r8_rows", 0x80FF80);

            const u32 dst_stride = ((width * 3 + 3) / 4) * 4;
            const u32 src_stride = width;

            for (u32 y = y_begin; y < y_end; ++y) {
                const u8 *s = src + (height - 1 - y) * src_stride;
                u8 *d = dst + y * dst_stride;

                for (u32 x = 0; x < width; ++x) {
                    u8 v = s[x];
                    d[0] = v;
                    d[1] = v;
                    d[2] = v;
                    d += 3;
                }
            }
        }

        // ============================
        // Dispatcher
        // ============================

        void convert_pixels_mt(std::vector<u8> &out, const u8 *pixel_data, u32 width, u32 height, VkFormat format) {
            ZoneScopedNC("convert_pixels_mt", 0xFFFFFF);

            init_srgb_lut();

            const u32 threads = std::max(1u, std::thread::hardware_concurrency());
            const u32 rows_per_thread = (height + threads - 1) / threads;

            std::vector<std::future<void>> futures;
            futures.reserve(threads);

            for (u32 t = 0; t < threads; ++t) {
                u32 y0 = t * rows_per_thread;
                u32 y1 = std::min(height, y0 + rows_per_thread);
                if (y0 >= y1)
                    break;

                futures.push_back(std::async(std::launch::async, [=, &out] {
                    switch (format) {
                        case VK_FORMAT_R8G8B8A8_UNORM:
                        case VK_FORMAT_R8G8B8A8_SRGB:
                            convert_rgba8_rows(out.data(), pixel_data, width, height, y0, y1);
                            break;

                        case VK_FORMAT_R32G32B32A32_SFLOAT:
                            convert_rgba32f_rows(out.data(), reinterpret_cast<const float *>(pixel_data), width, height,
                                                 y0, y1);
                            break;

                        case VK_FORMAT_R8_UNORM:
                        case VK_FORMAT_R8_SRGB:
                        case VK_FORMAT_R8_UINT:
                        case VK_FORMAT_R8_SINT:
                            convert_r8_rows(out.data(), pixel_data, width, height, y0, y1);
                            break;

                        default:
                            break;
                    }
                }));
            }

            for (auto &f: futures)
                f.get();
        }

    } // anonymous namespace

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
    };

    // Batch write multiple images
    void write_batch_to_disk(VmaAllocator &allocator, std::span<const ImageWriteRequest> requests) {
        ZoneScopedNC("write_batch_to_disk", 0x8050FF);

        if (requests.empty()) {
            return;
        }

        VmaAllocatorInfo allocator_info{};
        vmaGetAllocatorInfo(allocator, &allocator_info);

        // Create staging buffers for all images
        std::vector<StagingBuffer> staging_buffers;
        staging_buffers.reserve(requests.size());

        {
            ZoneScopedNC("create_staging_buffers", 0x4080FF);

            for (auto const &req: requests) {
                if (!req.texture) {
                    error("Null texture for {}", req.filename);
                    continue;
                }

                auto const &tex = *req.texture;

                u32 pixel_size = 0;
                switch (tex.format) {
                    case VK_FORMAT_R8G8B8A8_UNORM:
                    case VK_FORMAT_R8G8B8A8_SRGB:
                        pixel_size = 4;
                        break;
                    case VK_FORMAT_R32G32B32A32_SFLOAT:
                        pixel_size = 16;
                        break;
                    case VK_FORMAT_R8_UNORM:
                    case VK_FORMAT_R8_SRGB:
                    case VK_FORMAT_R8_UINT:
                    case VK_FORMAT_R8_SINT:
                        pixel_size = 1;
                        break;
                    default:
                        error("Unsupported format for {}: {}", req.filename, static_cast<u32>(tex.format));
                        continue;
                }

                VkDeviceSize buffer_size = static_cast<VkDeviceSize>(tex.width) * tex.height * pixel_size;

                VkBufferCreateInfo buffer_create_info{
                        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                        .size = buffer_size,
                        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                };

                VmaAllocationCreateInfo alloc_create_info{
                        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
                        .usage = VMA_MEMORY_USAGE_AUTO,
                };

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
                staging_buffers.push_back(staging);
            }
        }

        if (staging_buffers.empty()) {
            return;
        }

        // Create command pool and buffer
        VkCommandPool command_pool{};
        {
            ZoneScopedNC("create_command_pool", 0x4080FF);

            VkCommandPoolCreateInfo info{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                    .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                    .queueFamilyIndex = 0,
            };

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

            VkCommandBufferAllocateInfo info{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                    .commandPool = command_pool,
                    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    .commandBufferCount = 1,
            };

            if (vkAllocateCommandBuffers(allocator_info.device, &info, &command_buffer) != VK_SUCCESS) {
                error("Failed to allocate command buffer");
                vkDestroyCommandPool(allocator_info.device, command_pool, nullptr);
                for (auto const &sb: staging_buffers) {
                    vmaDestroyBuffer(allocator, sb.buffer, sb.allocation);
                }
                return;
            }
        }

        // Record all copy commands in a single command buffer
        {
            ZoneScopedNC("record_batch_commands", 0x40FFFF);

            VkCommandBufferBeginInfo begin_info{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };

            vkBeginCommandBuffer(command_buffer, &begin_info);

            for (size_t i = 0; i < requests.size(); ++i) {
                if (i >= staging_buffers.size())
                    break;

                auto const &tex = *requests[i].texture;
                auto const &staging = staging_buffers[i];

                // Transition to transfer source
                VkImageMemoryBarrier2 barrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        .image = tex.image,
                        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                };

                VkDependencyInfo dep{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                     .imageMemoryBarrierCount = 1,
                                     .pImageMemoryBarriers = &barrier};

                vkCmdPipelineBarrier2(command_buffer, &dep);

                // Copy image to buffer
                VkBufferImageCopy region{
                        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                        .imageExtent = {tex.width, tex.height, 1},
                };

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
            VkFenceCreateInfo info{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            vkCreateFence(allocator_info.device, &info, nullptr, &fence);

            VkQueue queue{};
            vkGetDeviceQueue(allocator_info.device, 0, 0, &queue);

            VkSubmitInfo submit{
                    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .commandBufferCount = 1,
                    .pCommandBuffers = &command_buffer,
            };

            vkQueueSubmit(queue, 1, &submit, fence);
            vkWaitForFences(allocator_info.device, 1, &fence, VK_TRUE, UINT64_MAX);

            vkDestroyFence(allocator_info.device, fence, nullptr);
        }

        // Now process all images in parallel on CPU
        {
            ZoneScopedNC("parallel_cpu_processing", 0x40FF40);

            std::vector<std::future<void>> futures;
            futures.reserve(staging_buffers.size());

            for (size_t i = 0; i < staging_buffers.size(); ++i) {
                futures.push_back(std::async(std::launch::async, [&, i]() {
                    ZoneScopedNC("process_single_image", 0xFF40FF);

                    auto const &staging = staging_buffers[i];
                    auto const &req = requests[i];

                    std::ofstream output(req.filename, std::ios::binary);
                    if (!output) {
                        error("Failed to open {}", req.filename);
                        return;
                    }

                    const u32 row_size = ((staging.width * 3 + 3) / 4) * 4;
                    std::vector<u8> cpu_pixels(row_size * staging.height);

                    const u8 *pixel_data = static_cast<const u8 *>(staging.alloc_info.pMappedData);

                    write_bmp_headers(output, staging.width, staging.height);
                    convert_pixels_mt(cpu_pixels, pixel_data, staging.width, staging.height, staging.format);

                    output.write(reinterpret_cast<char *>(cpu_pixels.data()), cpu_pixels.size());
                    output.close();
                }));
            }

            // Wait for all conversions to complete
            for (auto &f: futures) {
                f.get();
            }
        }

        // Cleanup
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

        std::ofstream output(filename.data(), std::ios::binary);
        if (!output) {
            error("Failed to open {}", filename);
            return;
        }

        const auto &tex = *texture;

        const u32 row_size = ((tex.width * 3 + 3) / 4) * 4;
        std::vector<u8> cpu_pixels(row_size * tex.height);

        u32 pixel_size = 0;
        {
            ZoneScopedNC("format_resolve", 0x4080FF);
            switch (tex.format) {
                case VK_FORMAT_R8G8B8A8_UNORM:
                case VK_FORMAT_R8G8B8A8_SRGB:
                    pixel_size = 4;
                    break;
                case VK_FORMAT_R32G32B32A32_SFLOAT:
                    pixel_size = 16;
                    break;
                case VK_FORMAT_R8_UNORM:
                case VK_FORMAT_R8_SRGB:
                case VK_FORMAT_R8_UINT:
                case VK_FORMAT_R8_SINT:
                    pixel_size = 1;
                    break;
                default:
                    error("Unsupported format for writing to disk: {}", static_cast<u32>(tex.format));
                    return;
            }
        }

        VmaAllocatorInfo allocator_info{};
        vmaGetAllocatorInfo(allocator, &allocator_info);

        const VkDeviceSize buffer_size = static_cast<VkDeviceSize>(tex.width) * tex.height * pixel_size;

        VkBuffer staging_buffer{};
        VmaAllocation staging_allocation{};
        VmaAllocationInfo staging_alloc_info{};

        {
            ZoneScopedNC("create_staging_buffer", 0x4080FF);

            VkBufferCreateInfo buffer_create_info{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = buffer_size,
                    .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };

            VmaAllocationCreateInfo alloc_create_info{
                    .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                    .usage = VMA_MEMORY_USAGE_AUTO,
            };

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

            VkCommandPoolCreateInfo info{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                    .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                    .queueFamilyIndex = 0,
            };

            if (vkCreateCommandPool(allocator_info.device, &info, nullptr, &command_pool) != VK_SUCCESS) {
                vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
                return;
            }
        }

        VkCommandBuffer command_buffer{};
        {
            ZoneScopedNC("allocate_command_buffer", 0x4080FF);

            VkCommandBufferAllocateInfo info{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                    .commandPool = command_pool,
                    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    .commandBufferCount = 1,
            };

            if (vkAllocateCommandBuffers(allocator_info.device, &info, &command_buffer) != VK_SUCCESS) {
                vkDestroyCommandPool(allocator_info.device, command_pool, nullptr);
                vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
                return;
            }
        }

        {
            ZoneScopedNC("record_commands", 0x40FFFF);

            VkCommandBufferBeginInfo begin_info{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };

            vkBeginCommandBuffer(command_buffer, &begin_info);

            VkImageMemoryBarrier2 barrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                    .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    .image = tex.image,
                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
            };

            VkDependencyInfo dep{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                 .imageMemoryBarrierCount = 1,
                                 .pImageMemoryBarriers = &barrier};

            vkCmdPipelineBarrier2(command_buffer, &dep);

            VkBufferImageCopy region{
                    .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                    .imageExtent = {tex.width, tex.height, 1},
            };

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
            VkFenceCreateInfo info{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            vkCreateFence(allocator_info.device, &info, nullptr, &fence);

            VkQueue queue{};
            vkGetDeviceQueue(allocator_info.device, 0, 0, &queue);

            VkSubmitInfo submit{
                    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .commandBufferCount = 1,
                    .pCommandBuffers = &command_buffer,
            };

            vkQueueSubmit(queue, 1, &submit, fence);
            vkWaitForFences(allocator_info.device, 1, &fence, VK_TRUE, UINT64_MAX);

            vkDestroyFence(allocator_info.device, fence, nullptr);
        }

        const u8 *pixel_data = static_cast<const u8 *>(staging_alloc_info.pMappedData);

        write_bmp_headers(output, tex.width, tex.height);
        convert_pixels_mt(cpu_pixels, pixel_data, tex.width, tex.height, tex.format);

        {
            ZoneScopedNC("disk_write", 0xFF00FF);
            output.write(reinterpret_cast<char *>(cpu_pixels.data()), cpu_pixels.size());
        }

        {
            ZoneScopedNC("cleanup", 0x808080);
            output.close();
            vkDestroyCommandPool(allocator_info.device, command_pool, nullptr);
            vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
        }
    }

} // namespace image_operations
