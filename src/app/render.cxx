#include "app/render.hxx"

#include "Mesh.hxx"

auto write_mesh_indirect(RenderContext &ctx, u32 frame_index, FrameIndirectWriter &w,
                         AlignedRingBuffer<VkDrawIndexedIndirectCommand> &cmd_ring,
                         AlignedRingBuffer<u32> &material_id_ring, const MeshData &mesh, u32 instance_count,
                         u32 first_instance) -> DrawRanges {
    const u32 total_submeshes = static_cast<u32>(mesh.submeshes.size());
    const u32 opaque_base = w.allocate(total_submeshes);

    std::vector<VkDrawIndexedIndirectCommand> opaque_cmds, alpha_cmds;
    std::vector<u32> opaque_mats, alpha_mats;

    for (const auto &s: mesh.submeshes) {
        VkDrawIndexedIndirectCommand c{
                .indexCount = s.index_count,
                .instanceCount = instance_count,
                .firstIndex = s.index_offset,
                .vertexOffset = 0,
                .firstInstance = first_instance,
        };

        if (s.alpha_tested) {
            alpha_cmds.push_back(c);
            alpha_mats.push_back(s.material_id);
        } else {
            opaque_cmds.push_back(c);
            opaque_mats.push_back(s.material_id);
        }
    }

    const u32 opaque_count = static_cast<u32>(opaque_cmds.size());
    const u32 alpha_count = static_cast<u32>(alpha_cmds.size());

    if (opaque_count > 0) {
        cmd_ring.write_elements(ctx, frame_index, opaque_base, std::span(opaque_cmds));
        material_id_ring.write_elements(ctx, frame_index, opaque_base, std::span(opaque_mats));
    }

    if (alpha_count > 0) {
        cmd_ring.write_elements(ctx, frame_index, opaque_base + opaque_count, std::span(alpha_cmds));
        material_id_ring.write_elements(ctx, frame_index, opaque_base + opaque_count, std::span(alpha_mats));
    }

    return {
            .opaque_base = opaque_base,
            .opaque_count = opaque_count,
            .alpha_base = opaque_base + opaque_count,
            .alpha_count = alpha_count,
    };
}
