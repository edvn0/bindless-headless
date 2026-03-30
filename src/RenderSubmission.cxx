#include "RenderSubmission.hxx"

#include "scene/Scene.hxx"

#include "scene/Components.hxx"

auto submit_mesh_instances(const Scene &scene, RenderQueue &queue) -> void {
    for (auto &&[e, mesh, xform]: scene.mesh_group.each()) {
        queue.submit({
                .transform = xform.local_to_world,
                .mesh_index = mesh.mesh_index,
                .lod_level = 0,
        });
    }
}
