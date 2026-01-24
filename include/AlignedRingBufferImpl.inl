#pragma once

#include <algorithm> // std::max
#include <cstring>   // std::memset

template<typename T, std::size_t N>
requires std::is_trivial_v<T>
auto AlignedRingBuffer<T, N>::create(RenderContext& ctx,
                                     VkBufferUsageFlags extra_usage,
                                     std::string_view name)
    -> tl::expected<AlignedRingBuffer, BufferCreateError>
{
constexpr auto align_up_pow2 = [](u64 value, u64 alignment) -> u64 {
    return (value + alignment - 1) & ~(alignment - 1);
};

    AlignedRingBuffer out{};

    const u64 slot_alignment = std::max<u64>(16, static_cast<u64>(alignof(T)));
    out.stride_bytes = align_up_pow2(static_cast<u64>(sizeof(T)), slot_alignment);

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
    out.slot_addresses[i] = DeviceAddress {base_address + i * out.stride_bytes};
}
    return out;
}