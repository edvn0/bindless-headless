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

struct DrawRange {
    u32 base;
    u32 count;
};

struct DrawRanges {
    DrawRange opaque;
    DrawRange alpha;
    DrawRange double_sided;
};

using MaterialBaseOffset = u32;

struct SubmeshMaterialOverride {
    u32 mesh_index;
    u32 submesh_index;
    u32 material_pool_index; // global pool index
};

struct IndirectWriteBuffers {
    FrameIndirectWriter &writer;
    AlignedRingBuffer<VkDrawIndexedIndirectCommand> &cmd_ring;
    AlignedRingBuffer<u32> &material_id_ring;
};

struct MeshDrawInfo {
    u32 mesh_index;
    u32 material_pool_base;
    u32 instance_count;
    u32 first_instance;
    std::span<const SubmeshMaterialOverride> overrides;
};

auto write_mesh_indirect(RenderContext &ctx, u32 frame_index, IndirectWriteBuffers &buffers, const MeshData &mesh,
                         const MeshDrawInfo &draw_info) -> DrawRanges;
auto reserve_light_volumes(RenderContext &, u32, FrameIndirectWriter &,
                           AlignedRingBuffer<VkDrawMeshTasksIndirectCommandEXT> &, AlignedRingBuffer<u32> &, u32)
        -> u32;
