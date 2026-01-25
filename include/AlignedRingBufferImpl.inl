#pragma once
#include <algorithm> // std::max
#include <cstring>   // std::memset

template<typename T, std::size_t N>
requires std::is_trivial_v<T>
auto AlignedRingBuffer<T, N>::create(RenderContext& ctx,
                                     u64 elements_per_slot,
                                     VkBufferUsageFlags extra_usage,
                                     std::string_view name)
    -> tl::expected<AlignedRingBuffer, BufferCreateError>
{
    constexpr auto align_up_pow2 = [](u64 value, u64 alignment) -> u64 {
        return (value + alignment - 1) & ~(alignment - 1);
    };
    
    if (elements_per_slot == 0) {
        error("AlignedRingBuffer: elements_per_slot must be > 0");
        return tl::unexpected{BufferCreateError{}};
    }
    
    AlignedRingBuffer out{};
    out.element_count = elements_per_slot;
    
    VmaAllocatorInfo alloc_info{};
    vmaGetAllocatorInfo(ctx.allocator, &alloc_info);
    VkPhysicalDeviceProperties pd_props{};
    vkGetPhysicalDeviceProperties(alloc_info.physicalDevice, &pd_props);
    
    // Align each element, then align the entire slot
    const u64 element_alignment = std::max<u64>(16, static_cast<u64>(alignof(T)));
    const u64 aligned_element_size = align_up_pow2(static_cast<u64>(sizeof(T)), element_alignment);
    const u64 slot_size = aligned_element_size * elements_per_slot;
    
    // Use actual device requirements instead of conservative 256
    const u64 min_storage_buffer_offset_alignment = 
        static_cast<u64>(pd_props.limits.minStorageBufferOffsetAlignment);
    const u64 slot_alignment = std::max<u64>(min_storage_buffer_offset_alignment, element_alignment);
    out.stride_bytes = align_up_pow2(slot_size, slot_alignment);
    
    const u64 total_bytes = out.stride_bytes * slot_count;
    
    auto buf = Buffer::zeroes(
        ctx.allocator,
        VkBufferCreateInfo{
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                     extra_usage,
        },
        VmaAllocationCreateInfo{},
        static_cast<std::size_t>(total_bytes),
        name
    );
    
    if (!buf) {
        return tl::unexpected{buf.error()};
    }
    
    out.buffer = ctx.create_buffer(std::move(*buf));
    const auto base_address = static_cast<u64>(ctx.device_address(out.buffer));
    
    for (u64 i = 0; i < slot_count; ++i) {
        out.slot_addresses[i] = DeviceAddress{base_address + i * out.stride_bytes};
    }
    
    return out;
}