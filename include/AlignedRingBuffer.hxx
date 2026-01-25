#pragma once
#include <ranges>
#include "Logger.hxx"
#include "RenderContext.hxx"

template<typename T, std::size_t N = frames_in_flight>
    requires std::is_trivial_v<T>
class AlignedRingBuffer {
    BufferHandle buffer{};
    u64 stride_bytes{0};
    u64 element_count{1}; // Number of T elements per slot
    std::array<DeviceAddress, N> slot_addresses{};

    auto base_address() const noexcept -> DeviceAddress { return slot_addresses[0]; }

public:
    static constexpr u64 slot_count{N};

    [[nodiscard]] auto handle() const noexcept -> BufferHandle { return buffer; }
    [[nodiscard]] auto stride() const noexcept -> u64 { return stride_bytes; }
    [[nodiscard]] auto elements_per_slot() const noexcept -> u64 { return element_count; }
    [[nodiscard]] auto slot_offset_bytes(u64 index) const noexcept -> u64 { return index * stride_bytes; }
    [[nodiscard]] auto slot_device_address(u64 index) const noexcept -> DeviceAddress { return slot_addresses[index]; }

    // Write a single element at element_index within the given slot
    auto write_element(RenderContext &ctx, u64 slot_index, u64 element_index, T const &value) -> void {
        auto *buf = ctx.buffers.get(buffer);
        if (!buf) {
            error("AlignedRingBuffer: invalid buffer handle");
            return;
        }
        if (element_index >= element_count) {
            error("AlignedRingBuffer: element_index {} out of bounds (max: {})", element_index, element_count);
            return;
        }
        const u64 offset = slot_offset_bytes(slot_index) + element_index * sizeof(T);
        buf->write_slice(ctx.allocator, std::span{&value, 1}, static_cast<std::size_t>(offset));
    }

    // Write multiple elements starting at element_index within the given slot
    auto write_elements(RenderContext &ctx, u64 slot_index, u64 element_index, std::span<T const> values) -> void {
        auto *buf = ctx.buffers.get(buffer);
        if (!buf) {
            error("AlignedRingBuffer: invalid buffer handle");
            return;
        }
        if (element_index + values.size() > element_count) {
            error("AlignedRingBuffer: write range [{}, {}) exceeds capacity {}", element_index,
                  element_index + values.size(), element_count);
            return;
        }
        const u64 offset = slot_offset_bytes(slot_index) + element_index * sizeof(T);
        buf->write_slice(ctx.allocator, values, static_cast<std::size_t>(offset));
    }

    // Write entire slot worth of elements
    auto write_slot(RenderContext &ctx, u64 slot_index, std::span<T const> values) -> void {
        if (values.size() != element_count) {
            error("AlignedRingBuffer: expected {} elements, got {}", element_count, values.size());
            return;
        }
        write_elements(ctx, slot_index, 0, values);
    }

    // Convenience overload for single-element slots (backwards compatibility)
    auto write_slot(RenderContext &ctx, u64 slot_index, T const &value) -> void
        requires(std::is_same_v<T, T>) // Always true, just for SFINAE
    {
        if (element_count != 1) {
            error("AlignedRingBuffer: write_slot(single) called on multi-element buffer");
            return;
        }
        write_element(ctx, slot_index, 0, value);
    }

    // Write same data to all slots (useful for initialization)
    auto write_all_slots(RenderContext &ctx, std::span<T const> values) -> void {
        if (values.size() != element_count) {
            error("AlignedRingBuffer: expected {} elements, got {}", element_count, values.size());
            return;
        }
        auto *buf = ctx.buffers.get(buffer);
        if (!buf) {
            error("AlignedRingBuffer: invalid buffer handle");
            return;
        }
        for (const auto slot_idx: std::views::iota(0uL, slot_count)) {
            buf->write_slice(ctx.allocator, values, static_cast<std::size_t>(slot_offset_bytes(slot_idx)));
        }
    }

    // Backwards compatibility for single-element slots
    auto write_all_slots(RenderContext &ctx, T const &value) -> void {
        if (element_count != 1) {
            error("AlignedRingBuffer: write_all_slots(single) called on multi-element buffer");
            return;
        }
        auto *buf = ctx.buffers.get(buffer);
        if (!buf) {
            error("AlignedRingBuffer: invalid buffer handle");
            return;
        }
        for (const auto slot_idx: std::views::iota(0uL, slot_count)) {
            buf->write_slice(ctx.allocator, std::span{&value, 1},
                             static_cast<std::size_t>(slot_offset_bytes(slot_idx)));
        }
    }

    template<typename FieldT>
        requires std::is_trivial_v<FieldT>
    auto write_field(RenderContext &ctx, u64 slot_index, u64 element_index, FieldT const &value, u64 field_offset_bytes)
            -> void {
        auto *buf = ctx.buffers.get(buffer);
        if (!buf) {
            error("AlignedRingBuffer: invalid buffer handle");
            return;
        }
        if (element_index >= element_count) {
            error("AlignedRingBuffer: element_index {} out of bounds", element_index);
            return;
        }
        const u64 base = slot_offset_bytes(slot_index) + element_index * sizeof(T);
        const u64 off = base + field_offset_bytes;
        buf->write_slice(ctx.allocator, std::span{&value, 1}, static_cast<std::size_t>(off));
    }

    // Backwards compatibility: write to first element's field
    template<typename FieldT>
        requires std::is_trivial_v<FieldT>
    auto write_field(RenderContext &ctx, u64 slot_index, FieldT const &value, u64 field_offset_bytes) -> void {
        write_field(ctx, slot_index, 0, value, field_offset_bytes);
    }

    // Create with runtime element count
    static auto create(RenderContext &ctx, u64 elements_per_slot, VkBufferUsageFlags extra_usage, std::string_view name)
            -> tl::expected<AlignedRingBuffer, BufferCreateError>;

    // Convenience: single element per slot
    static auto create(RenderContext &ctx, VkBufferUsageFlags extra_usage, std::string_view name)
            -> tl::expected<AlignedRingBuffer, BufferCreateError> {
        return create(ctx, 1, extra_usage, name);
    }

    static auto create(RenderContext &ctx, std::string_view name)
            -> tl::expected<AlignedRingBuffer, BufferCreateError> {
        return create(ctx, 1, VkBufferUsageFlags{0}, name);
    }
};

#include "AlignedRingBufferImpl.inl"
