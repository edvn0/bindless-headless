#include "../include/ImageOperations.hxx"
#include <cstring>

#include "Logger.hxx"

namespace image_operations {
    namespace {
        auto write_bmp_headers(std::ofstream &output, u32 width, u32 height) -> void {
            auto row_size = ((width * 3 + 3) / 4) * 4; // 3 bytes per pixel, padded to 4-byte boundary
            auto pixel_data_size = row_size * height;
            auto file_size = 14u + 40u + pixel_data_size; // file header + info header + pixels
            auto offset_data = 14u + 40u;

            u16 file_type = 0x4D42; // "BM"
            output.write(reinterpret_cast<char *>(&file_type), 2);
            output.write(reinterpret_cast<char *>(&file_size), 4);
            u16 reserved1 = 0;
            u16 reserved2 = 0;
            output.write(reinterpret_cast<char *>(&reserved1), 2);
            output.write(reinterpret_cast<char *>(&reserved2), 2);
            output.write(reinterpret_cast<char *>(&offset_data), 4);

            u32 dib_header_size = 40;
            i32 image_width = static_cast<i32>(width);
            i32 image_height = static_cast<i32>(height);
            u16 planes = 1;
            u16 bits_per_pixel = 24;
            u32 compression = 0; // BI_RGB
            u32 image_size = pixel_data_size;
            i32 x_pixels_per_meter = 0;
            i32 y_pixels_per_meter = 0;
            u32 colors_used = 0;
            u32 colors_important = 0;

            output.write(reinterpret_cast<char *>(&dib_header_size), 4);
            output.write(reinterpret_cast<char *>(&image_width), 4);
            output.write(reinterpret_cast<char *>(&image_height), 4);
            output.write(reinterpret_cast<char *>(&planes), 2);
            output.write(reinterpret_cast<char *>(&bits_per_pixel), 2);
            output.write(reinterpret_cast<char *>(&compression), 4);
            output.write(reinterpret_cast<char *>(&image_size), 4);
            output.write(reinterpret_cast<char *>(&x_pixels_per_meter), 4);
            output.write(reinterpret_cast<char *>(&y_pixels_per_meter), 4);
            output.write(reinterpret_cast<char *>(&colors_used), 4);
            output.write(reinterpret_cast<char *>(&colors_important), 4);
        }

        auto convert_and_write_pixels(std::ofstream &output, u8 const *pixel_data, u32 width, u32 height,
                                      VkFormat format) -> void {
            auto row_size = ((width * 3 + 3) / 4) * 4;
            auto padding = row_size - width * 3;
            auto padding_bytes = std::array<u8, 3>{0, 0, 0};

            for (i32 y = static_cast<i32>(height) - 1; y >= 0; --y) {
                for (u32 x = 0; x < width; ++x) {
                    auto pixel_idx = (y * width + x);
                    u8 bgr[3] = {0, 0, 0};

                    switch (format) {
                        case VK_FORMAT_R8G8B8A8_UNORM:
                        case VK_FORMAT_R8G8B8A8_SRGB: {
                            auto offset = pixel_idx * 4;
                            bgr[0] = pixel_data[offset + 2]; // B
                            bgr[1] = pixel_data[offset + 1]; // G
                            bgr[2] = pixel_data[offset + 0]; // R
                            break;
                        }
                        case VK_FORMAT_R32G32B32A32_SFLOAT: {
                            auto offset = pixel_idx * 16;
                            auto const *float_data = reinterpret_cast<float const *>(pixel_data + offset);
                            auto clamp_convert = [](float val) -> u8 {
                                if (val <= 0.0f)
                                    return 0;
                                if (val >= 1.0f)
                                    return 255;
                                return static_cast<u8>(val * 255.0f + 0.5f);
                            };
                            bgr[0] = clamp_convert(float_data[2]); // B
                            bgr[1] = clamp_convert(float_data[1]); // G
                            bgr[2] = clamp_convert(float_data[0]); // R
                            break;
                        }
                        case VK_FORMAT_R8_UNORM:
                        case VK_FORMAT_R8_SRGB:
                        case VK_FORMAT_R8_UINT:
                        case VK_FORMAT_R8_SINT: {
                            auto offset = pixel_idx;
                            auto v = pixel_data[offset];
                            bgr[0] = v;
                            bgr[1] = v;
                            bgr[2] = v;
                            break;
                        }
                        default:
                            break;
                    }

                    output.write(reinterpret_cast<char *>(bgr), 3);
                }

                // Write row padding
                if (padding > 0) {
                    output.write(reinterpret_cast<char *>(padding_bytes.data()), padding);
                }
            }
        }
    } // namespace

    auto write_to_disk(DestructionContext::TexturePool &textures, DestructionContext::TextureHandle texture,
                       VmaAllocator &allocator, std::string_view filename) -> void {
        auto output = std::ofstream{filename.data(), std::ios::binary};
        if (!output) {
            error("Failed to open output file for writing {}", filename);
            return;
        }

        auto resolved = textures.get(texture);
        if (!resolved) {
            error("Invalid texture handle for writing to disk");
            return;
        }

        auto const &tex = *resolved;

        auto pixel_size = 0u;
        switch (tex.format) {
            case VK_FORMAT_R8G8B8A8_UNORM:
            case VK_FORMAT_R8G8B8A8_SRGB:
                pixel_size = 4u;
                break;
            case VK_FORMAT_R32G32B32A32_SFLOAT:
                pixel_size = 16u;
                break;
            case VK_FORMAT_R8_UNORM:
            case VK_FORMAT_R8_SRGB:
            case VK_FORMAT_R8_UINT:
            case VK_FORMAT_R8_SINT:
                pixel_size = 1u;
                break;
            default:
                error("Unsupported format for writing to disk: {}", string_VkFormat(tex.format));
                return;
        }

        VmaAllocatorInfo allocator_info{};
        vmaGetAllocatorInfo(allocator, &allocator_info);

        auto buffer_size = static_cast<VkDeviceSize>(tex.width) * tex.height * pixel_size;

        VkBufferCreateInfo buffer_create_info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                              .size = buffer_size,
                                              .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                              .sharingMode = VK_SHARING_MODE_EXCLUSIVE};

        VmaAllocationCreateInfo alloc_create_info{.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                                           VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                                  .usage = VMA_MEMORY_USAGE_AUTO};

        VkBuffer staging_buffer;
        VmaAllocation staging_allocation;
        VmaAllocationInfo staging_alloc_info;

        auto result = vmaCreateBuffer(allocator, &buffer_create_info, &alloc_create_info, &staging_buffer,
                                      &staging_allocation, &staging_alloc_info);
        if (result != VK_SUCCESS) {
            error("Failed to create staging buffer: {}", string_VkResult(result));
            return;
        }

        auto command_pool = VkCommandPool{};
        VkCommandPoolCreateInfo command_pool_create_info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                .queueFamilyIndex = 0,
        };
        result = vkCreateCommandPool(allocator_info.device, &command_pool_create_info, nullptr, &command_pool);
        if (result != VK_SUCCESS) {
            error("Failed to create command pool: {}", string_VkResult(result));
            vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
            return;
        }

        auto command_buffer = VkCommandBuffer{};
        VkCommandBufferAllocateInfo command_buffer_alloc_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                                              .commandPool = command_pool,
                                                              .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                                              .commandBufferCount = 1};
        result = vkAllocateCommandBuffers(allocator_info.device, &command_buffer_alloc_info, &command_buffer);
        if (result != VK_SUCCESS) {
            error("Failed to allocate command buffer: {}", string_VkResult(result));
            vkDestroyCommandPool(allocator_info.device, command_pool, nullptr);
            vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
            return;
        }

        VkCommandBufferBeginInfo command_buffer_begin_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                                           .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        vk_check(vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info));

        VkImageMemoryBarrier2 image_barrier{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                            .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                            .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                                            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                            .image = tex.image,
                                            .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                                 .baseMipLevel = 0,
                                                                 .levelCount = 1,
                                                                 .baseArrayLayer = 0,
                                                                 .layerCount = 1}};

        VkDependencyInfo dependency_info{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                         .imageMemoryBarrierCount = 1,
                                         .pImageMemoryBarriers = &image_barrier};

        vkCmdPipelineBarrier2(command_buffer, &dependency_info);

        VkBufferImageCopy region{.bufferOffset = 0,
                                 .bufferRowLength = 0,
                                 .bufferImageHeight = 0,
                                 .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                      .mipLevel = 0,
                                                      .baseArrayLayer = 0,
                                                      .layerCount = 1},
                                 .imageOffset = {0, 0, 0},
                                 .imageExtent = {tex.width, tex.height, 1}};

        vkCmdCopyImageToBuffer(command_buffer, tex.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging_buffer, 1,
                               &region);

        image_barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        image_barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        image_barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        image_barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        image_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        image_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;

        vkCmdPipelineBarrier2(command_buffer, &dependency_info);

        vk_check(vkEndCommandBuffer(command_buffer));

        VkFenceCreateInfo fence_create_info{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence;
        vk_check(vkCreateFence(allocator_info.device, &fence_create_info, nullptr, &fence));

        VkSubmitInfo submit_info{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &command_buffer};

        VkQueue queue;
        vkGetDeviceQueue(allocator_info.device, 0, 0, &queue);

        vk_check(vkQueueSubmit(queue, 1, &submit_info, fence));
        vk_check(vkWaitForFences(allocator_info.device, 1, &fence, VK_TRUE, UINT64_MAX));

        auto const *pixel_data = static_cast<u8 const *>(staging_alloc_info.pMappedData);

        write_bmp_headers(output, tex.width, tex.height);
        convert_and_write_pixels(output, pixel_data, tex.width, tex.height, tex.format);

        output.close();

        vkDestroyFence(allocator_info.device, fence, nullptr);
        vkDestroyCommandPool(allocator_info.device, command_pool, nullptr);
        vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
    }
} // namespace image_operations
