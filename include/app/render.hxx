#pragma once

#include "AlignedRingBuffer.hxx"
#include "Types.hxx"

struct FrameIndirectWriter {
    u32 cursor{0}; // in commands, not bytes

    auto allocate(u32 count) -> u32 {
        u32 base = cursor;
        cursor += count;
        return base;
    }
};

struct DrawRanges {
    u32 opaque_base;
    u32 opaque_count;
    u32 alpha_base;
    u32 alpha_count;
};

auto write_mesh_indirect(RenderContext &ctx, u32 frame_index, FrameIndirectWriter &w,
                         AlignedRingBuffer<VkDrawIndexedIndirectCommand> &cmd_ring,
                         AlignedRingBuffer<u32> &material_id_ring, const MeshData &mesh, u32 instance_count,
                         u32 first_instance) -> DrawRanges;
auto reserve_light_volumes(RenderContext &, u32, FrameIndirectWriter &,
                           AlignedRingBuffer<VkDrawMeshTasksIndirectCommandEXT> &, AlignedRingBuffer<u32> &, u32)
        -> u32;
