#pragma once

#include <ranges>

#include "Logger.hxx"
#include "RenderContext.hxx"

template<typename T, std::size_t N = frames_in_flight>
class AlignedRingBuffer {
    BufferHandle buffer{};
    u64 stride_bytes{0};
    u64 element_count{1}; // Number of T elements per slot
    std::array<DeviceAddress, N> slot_addresses{};

    [[nodiscard]] auto base_address() const noexcept -> DeviceAddress { return slot_addresses[0]; }

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

    auto write_all_slots(RenderContext &ctx, const T &value) -> void {
        auto *buf = ctx.buffers.get(buffer);
        if (!buf) {
            error("AlignedRingBuffer: invalid buffer handle");
            return;
        }

        auto result = buf->memset(ctx.allocator, value);
        if (!result) {
            error("AlignedRingBuffer: failed to memset buffer: {}", result.error().message);
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
        buf->write_slice(ctx.allocator, std::span{&value, 1}, off);
    }

    auto fill_zeros(VkCommandBuffer cmd, RenderContext &ctx, u32 slot_index) -> void {
        auto *buf = ctx.buffers.get(handle());
        vkCmdFillBuffer(cmd, buf->buffer(), slot_offset_bytes(slot_index), stride(), 0);

        auto barrier = create_info<VkMemoryBarrier2>();

        barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        auto dep = create_info<VkDependencyInfo>();
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &barrier;

        vkCmdPipelineBarrier2(cmd, &dep);
    }

    template<typename FieldT>
        requires std::is_trivial_v<FieldT>
    auto write_field(RenderContext &ctx, u64 slot_index, FieldT const &value, u64 field_offset_bytes) -> void {
        write_field(ctx, slot_index, 0, value, field_offset_bytes);
    }

    static auto create(RenderContext &ctx, u64 elements_per_slot, VkBufferUsageFlags extra_usage, std::string_view name, const std::span<const u32> queue_indices = {})
            -> tl::expected<AlignedRingBuffer, Error>;

    static auto create(RenderContext &ctx, VkBufferUsageFlags extra_usage, std::string_view name, const std::span<const u32> queue_indices = {})
            -> tl::expected<AlignedRingBuffer, Error> {
        return create(ctx, 1, extra_usage, name, queue_indices);
    }

    static auto create(RenderContext &ctx, std::string_view name, const std::span<const u32> queue_indices = {}) -> tl::expected<AlignedRingBuffer, Error> {
        return create(ctx, 1, VkBufferUsageFlags{0}, name, queue_indices);
    }

    static auto recreate(RenderContext& ctx, u64 retire_value, AlignedRingBuffer& current, u64 new_element_count, std::string_view name, const std::span<const u32> queue_indices = {}) -> void {
    auto old_buffer_handle = current.handle();

    auto new_buffer = AlignedRingBuffer<T, N>::create(ctx, new_element_count, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, name, queue_indices);
    
    if (new_buffer) {
        current = std::move(*new_buffer);
    } else {
        error("Failed to recreate AlignedRingBuffer: {}", name);
        return;
    }

    destroy(ctx, old_buffer_handle, retire_value);
}
};



#include "AlignedRingBufferImpl.inl"
