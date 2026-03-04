#include "RenderSubmission.hxx"

#include "scene/Scene.hxx"

#include "scene/Components.hxx"

auto submit_mesh_instances(Scene &scene, RenderQueue &queue) -> void {
    scene.registry.sort<MeshComponent>([](const auto& lhs, const auto& rhs) {
        return lhs.mesh_index < rhs.mesh_index;
    });
    auto view = scene.registry.view<MeshComponent, TransformComponent>();
    for (auto&& [e, mesh, xform] : view.each()) {
        queue.submit({
            .mesh_index  = mesh.mesh_index,
            .transform   = xform.local_to_world,
            .material_id = 0, // TODO: Material ID on mesh or material component or something
        });
    }
}