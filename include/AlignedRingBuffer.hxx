#pragma once

#include <ranges>
#include "RenderContext.hxx"
#include "Logger.hxx"

template<typename T, std::size_t N = frames_in_flight>
requires std::is_trivial_v<T>
class AlignedRingBuffer {
    BufferHandle buffer{};
    u64 stride_bytes{0};
std::array<DeviceAddress, N> slot_addresses{};
    auto base_address() const noexcept -> DeviceAddress { return slot_addresses[0]; }

public:
    static constexpr u64 slot_count{N};

    [[nodiscard]] auto handle() const noexcept -> BufferHandle { return buffer; }
    [[nodiscard]] auto stride() const noexcept -> u64 { return stride_bytes; }

    [[nodiscard]] auto slot_offset_bytes(u64 index) const noexcept -> u64 { return index * stride_bytes; }

    [[nodiscard]] auto slot_device_address(u64 index) const noexcept -> DeviceAddress {
        return slot_addresses[index];
    }

    auto write_slot(RenderContext& ctx, u64 index, T const& value) -> void {
        auto* buf = ctx.buffers.get(buffer);
        if (!buf) {
            error("AlignedRingBuffer: invalid buffer handle");
            return;
        }
        buf->write_slice(ctx.allocator, std::span{&value, 1}, static_cast<std::size_t>(slot_offset_bytes(index)));
    }

    // Future me - this needs to be a memcpy over the entire buffer. 
    // But technically, we will never need to resize a N-sized ringbuffer in a hot-path, 
    // which means this code will probably only run on resizes etc. 
    auto write_all_slots(RenderContext& ctx, T const& value) -> void {
        auto* buf = ctx.buffers.get(buffer);
        if (!buf) {
            error("AlignedRingBuffer: invalid buffer handle");
            return;
        }
        for (const auto val: std::views::iota(0uL, slot_count)) {
            buf->write_slice(ctx.allocator, std::span{&value, 1}, static_cast<std::size_t>(slot_offset_bytes(val)));
        }
    }

template<typename FieldT>
requires std::is_trivial_v<FieldT>
auto write_field(RenderContext& ctx,
                                          u64 slot_index,
                                          FieldT const& value,
                                          u64 field_offset_bytes) -> void
{
    auto* buf = ctx.buffers.get(buffer);
    if (!buf) {
        error("AlignedRingBuffer: invalid buffer handle");
        return;
    }

    const u64 base = slot_offset_bytes(slot_index);
    const u64 off  = base + field_offset_bytes;

    buf->write_slice(ctx.allocator,
                     std::span{&value, 1},
                     static_cast<std::size_t>(off));
}

    static auto create(RenderContext& ctx,
                       VkBufferUsageFlags extra_usage,
                       std::string_view name) -> tl::expected<AlignedRingBuffer, BufferCreateError>;
                       
    static auto create(RenderContext& ctx,
                       std::string_view name) -> tl::expected<AlignedRingBuffer, BufferCreateError> {
        return create(ctx, VkBufferUsageFlags{0}, name);
                       }
};

#include "AlignedRingBufferImpl.inl"