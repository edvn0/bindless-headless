#pragma once

#include <cstring>
#include <optional>
#include <span>
#include <tl/expected.hpp>
#include <type_traits>

#include "Assert.hxx"
#include "CreateInfo.hxx"
#include "Logger.hxx"
#include "Types.hxx"

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
        requires std::is_trivially_copyable_v<T>
    auto read_into_slice(VmaAllocator &alloc, std::span<T, N> slice, std::size_t offset = 0) const -> void {
        auto *data = allocation_info.pMappedData;
        if (!data) {
            error("Trying to read from non-mapped memory. How?");
            return;
        }

        if (offset + slice.size_bytes() > size()) {
            error("Trying to read out of bounds memory");
            return;
        }
        vmaInvalidateAllocation(alloc, allocation(), static_cast<VkDeviceSize>(offset),
                                static_cast<VkDeviceSize>(slice.size_bytes()));
        const auto offset_data = static_cast<u8 *>(data) + offset;
        std::memcpy(slice.data(), offset_data, slice.size_bytes());
    }

    template<typename T, std::size_t N = std::dynamic_extent>
        requires std::is_trivially_copyable_v<T>
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
        requires std::is_trivially_copyable_v<T>
    auto write_slice(VmaAllocator &alloc, std::span<const T, N> slice, std::size_t offset = 0) -> void {
        auto *data = allocation_info.pMappedData;
        if (!data) {
            error("Trying to write into non-mapped memory. How?");
            return;
        }
        if (offset + slice.size_bytes() > size()) {
            error("Trying to overwrite memory");
            return;
        }
        auto *offset_data = static_cast<u8 *>(data) + offset;
        std::memcpy(offset_data, slice.data(), slice.size_bytes());
        vk_check(vmaFlushAllocation(alloc, allocation(), static_cast<VkDeviceSize>(offset),
                                    static_cast<VkDeviceSize>(slice.size_bytes())));
    }

    template<typename T>
        requires std::is_trivially_copyable_v<T>
    static auto from_slice(VmaAllocator &allocator, VkBufferUsageFlags usage_flags, const std::span<const T> slice,
                           const std::string_view name, const std::span<const u32> queue_indices = {})
            -> tl::expected<Buffer, Error> {
        const auto size = slice.size_bytes();

        // Get physical device alignment requirements
        VmaAllocatorInfo alloc_info{};
        vmaGetAllocatorInfo(allocator, &alloc_info);
        VkPhysicalDeviceProperties pd_props{};
        vkGetPhysicalDeviceProperties(alloc_info.physicalDevice, &pd_props);

        // Align size to device requirements
        const auto min_alignment = static_cast<u64>(pd_props.limits.minStorageBufferOffsetAlignment);
        const auto aligned_size = (size + min_alignment - 1) & ~(min_alignment - 1);

        auto ci = create_info<VkBufferCreateInfo>();
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.pNext = nullptr;
        ci.flags = 0;
        ci.size = aligned_size;
        ci.usage = usage_flags | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // Just for clarity here. 0 = EXCLUSIVE.
        if (!queue_indices.empty()) {
            ci.sharingMode = VK_SHARING_MODE_CONCURRENT;
            ci.queueFamilyIndexCount = static_cast<u32>(queue_indices.size());
            ci.pQueueFamilyIndices = queue_indices.data();
        }

        auto ai = create_info<VmaAllocationCreateInfo>();
        ai.usage = VMA_MEMORY_USAGE_AUTO;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        ai.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

        Buffer buffer{};
        if (const auto could = vmaCreateBuffer(allocator, &ci, &ai, &buffer.vk_buffer, &buffer.vma_allocation,
                                               &buffer.allocation_info);
            could != VK_SUCCESS) {
            return tl::unexpected{Error{.type = Error::Type::InvalidSize, .message = "Could not create buffer."}};
        }

        buffer.count = slice.size();
        buffer.set_name(allocator, name);

        auto dba_info = create_info<VkBufferDeviceAddressInfo>();
        dba_info.buffer = buffer.vk_buffer;

        buffer.dev_address = static_cast<DeviceAddress>(vkGetBufferDeviceAddress(alloc_info.device, &dba_info));
        ASSERT(buffer.dev_address != DeviceAddress::Invalid,
               "Buffer device address is invalid. Does the device support buffer device addresses?");

        const auto pointer = buffer.allocation_info.pMappedData;
        if (!pointer) {
            return tl::unexpected{Error{Error::Type::CouldNotMapMemory, "Buffer could not or was not mapped."}};
        }
        std::memcpy(pointer, slice.data(), slice.size_bytes());
        vk_check(vmaFlushAllocation(allocator, buffer.allocation(), 0, VK_WHOLE_SIZE));

        return buffer;
    }

    template<typename T>
        requires std::is_trivially_copyable_v<T>
    static auto from_value(VmaAllocator &allocator, VkBufferUsageFlags ci, const T &value, const std::string_view name)
            -> tl::expected<Buffer, Error> {
        return from_slice<T>(allocator, ci, std::span{&value, 1}, name);
    }

    static auto zeroes(VmaAllocator &allocator, VkBufferUsageFlags usage_flags, const std::size_t size,
                       const std::string_view name, const std::span<const u32> queue_indices = {})
            -> tl::expected<Buffer, Error> {
        // Get physical device alignment requirements
        VmaAllocatorInfo alloc_info{};
        vmaGetAllocatorInfo(allocator, &alloc_info);
        VkPhysicalDeviceProperties pd_props{};
        vkGetPhysicalDeviceProperties(alloc_info.physicalDevice, &pd_props);

        // Align size to device requirements
        const auto min_alignment = static_cast<u64>(pd_props.limits.minStorageBufferOffsetAlignment);
        const auto aligned_size = (size + min_alignment - 1) & ~(min_alignment - 1);

        VkBufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.pNext = nullptr;
        ci.flags = 0;
        ci.size = aligned_size;
        ci.usage = usage_flags | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

        if (!queue_indices.empty()) {
            ci.sharingMode = VK_SHARING_MODE_CONCURRENT;
            ci.queueFamilyIndexCount = static_cast<u32>(queue_indices.size());
            ci.pQueueFamilyIndices = queue_indices.data();
        } else {
            ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        ai.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

        Buffer buffer{};
        if (const auto could = vmaCreateBuffer(allocator, &ci, &ai, &buffer.vk_buffer, &buffer.vma_allocation,
                                               &buffer.allocation_info);
            could != VK_SUCCESS) {
            return tl::unexpected{Error{Error::Type::InvalidSize, "Size is invalid."}};
        }

        buffer.set_name(allocator, name);

        VkBufferDeviceAddressInfo dba_info{
                .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                .pNext = nullptr,
                .buffer = buffer.vk_buffer,
        };

        buffer.dev_address = static_cast<DeviceAddress>(vkGetBufferDeviceAddress(alloc_info.device, &dba_info));

        const auto pointer = buffer.allocation_info.pMappedData;
        if (!pointer) {
            return tl::unexpected{
                    Error{.type = Error::Type::CouldNotMapMemory, .message = "Buffer was not or could not be mapped."}};
        }
        std::memset(pointer, 0, aligned_size);
        vk_check(vmaFlushAllocation(allocator, buffer.allocation(), 0, VK_WHOLE_SIZE));

        return buffer;
    }

    [[nodiscard]] auto data_pointer() const { return allocation_info.pMappedData; }

    template<typename T>
        requires std::is_trivially_copyable_v<T>
    auto memset(VmaAllocator &allocator, const T &value) -> tl::expected<void, Error> {
        ASSERT(size() % sizeof(T) == 0, "Value is not aligned with the size of this buffer.");

        const auto pointer = data_pointer();
        if (!pointer) {
            return tl::unexpected{
                    Error{.type = Error::Type::CouldNotMapMemory, .message = "Buffer was not or could not be mapped."}};
        }

        auto *typed_pointer = static_cast<T *>(pointer);
        const auto element_count = size() / sizeof(T);

        for (std::size_t i = 0; i < element_count; ++i) {
            typed_pointer[i] = value;
        }

        vk_check(vmaFlushAllocation(allocator, vma_allocation, 0, VK_WHOLE_SIZE));

        return {};
    }

private:
    auto set_name(VmaAllocator &, std::string_view) const -> void;
};
