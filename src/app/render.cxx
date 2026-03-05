#include "app/render.hxx"
#include "Mesh.hxx"

auto write_mesh_indirect(RenderContext &ctx, u32 frame_index, IndirectWriteBuffers buffers, const MeshData &mesh,
                         const MeshDrawInfo &draw_info) -> DrawRanges {
    const u32 total_submeshes = static_cast<u32>(mesh.submeshes.size());
    const u32 opaque_base = buffers.writer.allocate(total_submeshes);

    std::vector<VkDrawIndexedIndirectCommand> opaque_cmds, alpha_cmds;
    std::vector<u32> opaque_mats, alpha_mats;

    for (u32 si = 0; si < mesh.submeshes.size(); ++si) {
        const auto &s = mesh.submeshes[si];

        VkDrawIndexedIndirectCommand c{
                .indexCount = s.index_count,
                .instanceCount = draw_info.instance_count,
                .firstIndex = s.index_offset,
                .vertexOffset = 0,
                .firstInstance = draw_info.first_instance,
        };

        u32 mat_id = draw_info.material_pool_base + s.material_id;
        for (const auto &ov: draw_info.overrides) {
            if (ov.mesh_index == draw_info.mesh_index && ov.submesh_index == si) {
                mat_id = ov.material_pool_index;
                break;
            }
        }

        if (s.alpha_tested) {
            alpha_cmds.push_back(c);
            alpha_mats.push_back(mat_id);
        } else {
            opaque_cmds.push_back(c);
            opaque_mats.push_back(mat_id);
        }
    }

    const u32 opaque_count = static_cast<u32>(opaque_cmds.size());
    const u32 alpha_count = static_cast<u32>(alpha_cmds.size());

    if (opaque_count > 0) {
        buffers.cmd_ring.write_elements(ctx, frame_index, opaque_base, std::span(opaque_cmds));
        buffers.material_id_ring.write_elements(ctx, frame_index, opaque_base, std::span(opaque_mats));
    }

    if (alpha_count > 0) {
        buffers.cmd_ring.write_elements(ctx, frame_index, opaque_base + opaque_count, std::span(alpha_cmds));
        buffers.material_id_ring.write_elements(ctx, frame_index, opaque_base + opaque_count, std::span(alpha_mats));
    }

    return {
            .opaque_base = opaque_base,
            .opaque_count = opaque_count,
            .alpha_base = opaque_base + opaque_count,
            .alpha_count = alpha_count,
    };
}

auto reserve_light_volumes(RenderContext &ctx, u32 frame_index, FrameIndirectWriter &w,
                           AlignedRingBuffer<VkDrawMeshTasksIndirectCommandEXT> &mesh_cmd_ring,
                           AlignedRingBuffer<u32> &material_id_ring, u32 light_material_id) -> u32 {
    const u32 slot = w.allocate(1);

    VkDrawMeshTasksIndirectCommandEXT light_cmd{
            .groupCountX = 0,
            .groupCountY = 1,
            .groupCountZ = 1,
    };

    mesh_cmd_ring.write_elements(ctx, frame_index, slot, {&light_cmd, 1});
    material_id_ring.write_elements(ctx, frame_index, slot, {&light_material_id, 1});

    return slot;
}
