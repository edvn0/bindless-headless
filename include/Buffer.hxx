#pragma once

#include <expected>
#include <span>
#include <type_traits>
#include <vma/vk_mem_alloc.h>

#include "Types.hxx"

struct BufferCreateError {
    enum class Type {
        InvalidSize = 0,
    };

    Type type{};
};

enum class DeviceAddress: std::uint64_t {
    Invalid = 0,
};

class Buffer {
    std::optional<u64> count;
    DeviceAddress dev_address{UINT64_MAX};
    VkBuffer vk_buffer{nullptr};
    VmaAllocation vma_allocation{nullptr};
    VmaAllocationInfo allocation_info{};
public:
    [[nodiscard]] auto size() const noexcept { return allocation_info.size; }
    [[nodiscard]] auto device_address() const noexcept { return dev_address; }
    [[nodiscard]] auto buffer() const noexcept { return vk_buffer; }
    [[nodiscard]] auto allocation() const noexcept { return vma_allocation; }

    template<typename T> requires std::is_trivial_v<T>
    static auto from_slice(VmaAllocator &allocator, VkBufferCreateInfo ci, VmaAllocationCreateInfo ai,
                           const std::span<const T> slice,
                           const std::string_view name) -> std::expected<Buffer, BufferCreateError> {
        const auto size = slice.size_bytes();
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.pNext = nullptr;
        ci.flags = 0;
        ci.size = size;
        ci.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        ai.usage        = VMA_MEMORY_USAGE_AUTO;
        ai.flags        = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                          VMA_ALLOCATION_CREATE_MAPPED_BIT;
        ai.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

        Buffer buffer{};
        if (const auto could = vmaCreateBuffer(allocator, &ci, &ai, &buffer.vk_buffer, &buffer.vma_allocation,
                                               &buffer.allocation_info); could != VK_SUCCESS) {
            return std::unexpected{BufferCreateError{BufferCreateError::Type::InvalidSize}};
        }

        buffer.count = slice.size();
        buffer.set_name(allocator, name);

        VmaAllocatorInfo info{};
        vmaGetAllocatorInfo(allocator, &info);

        VkBufferDeviceAddressInfo dba_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .pNext = nullptr,
            .buffer = buffer.vk_buffer,
        };

        buffer.dev_address = static_cast<DeviceAddress>(vkGetBufferDeviceAddress(info.device, &dba_info));

        const auto pointer = buffer.allocation_info.pMappedData;
        std::memcpy(pointer, slice.data(), slice.size_bytes());

        return buffer;
    }

private:
    auto set_name(VmaAllocator &, std::string_view) const -> void;
};
