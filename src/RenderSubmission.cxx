#include "RenderSubmission.hxx"

#include "scene/Scene.hxx"

#include "scene/Components.hxx"

auto submit_mesh_instances(Scene &scene, RenderQueue &queue) -> void {
    scene.registry.sort<MeshComponent>(
            [](const auto &lhs, const auto &rhs) { return lhs.mesh_index < rhs.mesh_index; });
    auto view = scene.registry.view<MeshComponent, TransformComponent>();
    for (auto &&[e, mesh, xform]: view.each()) {
        queue.submit({
                .transform = xform.local_to_world,
                .mesh_index = mesh.mesh_index,
                .lod_level = 0,
        });
    }
}
