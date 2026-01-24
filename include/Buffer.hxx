#pragma once

#include <cstring>
#include <optional>
#include <span>
#include <tl/expected.hpp>
#include <type_traits>


#include "Logger.hxx"
#include "Types.hxx"

struct BufferCreateError {
    enum class Type : u64 {
        InvalidSize = 0,
        CouldNotMapMemory = 1,
    };

    Type type{};
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
    [[nodiscard]] auto get_count() const noexcept -> u64 { return count.value_or(0); }

    template<typename T, std::size_t N = std::dynamic_extent>
        requires std::is_trivial_v<T>
    auto write_slice(VmaAllocator &alloc, std::span<T, N> slice, std::size_t offset = 0) {
        auto *data = allocation_info.pMappedData;
        if (!data) {
            error("Trying to write into non-mapped memory. How?");
            return;
        }
        if (offset + slice.size_bytes() > size()) {
            error("Trying to overwrite memory");
            return;
        }
        const auto offset_data = static_cast<u8 *>(data) + offset;
        std::memcpy(offset_data, slice.data(), slice.size_bytes());
        vk_check(vmaFlushAllocation(alloc, allocation(), static_cast<VkDeviceSize>(offset),
                                    static_cast<VkDeviceSize>(slice.size_bytes())));
    }

 template<typename T, std::size_t N = std::dynamic_extent>
    requires std::is_trivial_v<T>
auto write_slice(VmaAllocator& alloc, std::span<const T, N> slice, std::size_t offset = 0) -> void {
    auto* data = allocation_info.pMappedData;
    if (!data) {
        error("Trying to write into non-mapped memory. How?");
        return;
    }
    if (offset + slice.size_bytes() > size()) {
        error("Trying to overwrite memory");
        return;
    }
    auto* offset_data = static_cast<u8*>(data) + offset;
    std::memcpy(offset_data, slice.data(), slice.size_bytes());
    vk_check(vmaFlushAllocation(alloc,
                                allocation(),
                                static_cast<VkDeviceSize>(offset),
                                static_cast<VkDeviceSize>(slice.size_bytes())));
}

    template<typename T>
        requires std::is_trivial_v<T>
    static auto from_slice(VmaAllocator &allocator, VkBufferCreateInfo ci, VmaAllocationCreateInfo ai,
                           const std::span<const T> slice, const std::string_view name)
            -> tl::expected<Buffer, BufferCreateError> {
        const auto size = slice.size_bytes();
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.pNext = nullptr;
        ci.flags = 0;
        ci.size = size;
        ci.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

        ai.usage = VMA_MEMORY_USAGE_AUTO;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        ai.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

        Buffer buffer{};
        if (const auto could = vmaCreateBuffer(allocator, &ci, &ai, &buffer.vk_buffer, &buffer.vma_allocation,
                                               &buffer.allocation_info);
            could != VK_SUCCESS) {
            return tl::unexpected{BufferCreateError{BufferCreateError::Type::InvalidSize}};
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
        if (!pointer) {
    return tl::unexpected{BufferCreateError{BufferCreateError::Type::CouldNotMapMemory}};
}
        std::memcpy(pointer, slice.data(), slice.size_bytes());
        vk_check(vmaFlushAllocation(allocator, buffer.allocation(), 0, VK_WHOLE_SIZE));

        return buffer;
    }

    template<typename T>
        requires std::is_trivial_v<T>
    static auto from_value(VmaAllocator &allocator, VkBufferCreateInfo ci, VmaAllocationCreateInfo ai, const T &value,
                           const std::string_view name) -> tl::expected<Buffer, BufferCreateError> {
        return from_slice<T>(allocator, ci, ai, std::span{&value, 1}, name);
    }

    static auto zeroes(VmaAllocator &allocator, VkBufferCreateInfo ci, VmaAllocationCreateInfo ai,
                       const std::size_t size, const std::string_view name) -> tl::expected<Buffer, BufferCreateError> {
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.pNext = nullptr;
        ci.flags = 0;
        ci.size = size;
        ci.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

        ai.usage = VMA_MEMORY_USAGE_AUTO;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        ai.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

        Buffer buffer{};
        if (const auto could = vmaCreateBuffer(allocator, &ci, &ai, &buffer.vk_buffer, &buffer.vma_allocation,
                                               &buffer.allocation_info);
            could != VK_SUCCESS) {
            return tl::unexpected{BufferCreateError{BufferCreateError::Type::InvalidSize}};
        }

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
        if (!pointer) {
    return tl::unexpected{BufferCreateError{BufferCreateError::Type::CouldNotMapMemory}};
}
        std::memset(pointer, 0, size);
        vk_check(vmaFlushAllocation(allocator, buffer.allocation(), 0, VK_WHOLE_SIZE));

        return buffer;
    }

private:
    auto set_name(VmaAllocator &, std::string_view) const -> void;
};
