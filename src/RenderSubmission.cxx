#include "RenderSubmission.hxx"

#include "scene/Scene.hxx"

#include "scene/Components.hxx"

auto submit_mesh_instances(const Scene &scene, RenderQueue &queue) -> void {
    constexpr auto resolve_world = [](this auto &&self, const Scene &scene, entt::entity e) -> glm::mat4 {
        const auto &reg = scene.registry;

        if (e == entt::null || !reg.valid(e))
            return glm::mat4{1.0f};

        const auto local = glm::mat4{reg.get<TransformComponent>(e).local_to_world};

        if (reg.all_of<HierarchyComponent>(e)) {
            const auto &hc = reg.get<HierarchyComponent>(e);
            if (hc.parent != entt::null && reg.valid(hc.parent)) {
                return self(scene, hc.parent) * local;
            }
        }

        return local;
    };

    for (auto &&[e, mesh, xform]: scene.mesh_group.each()) {
        queue.submit({
                .transform = resolve_world(scene, e),
                .mesh_index = mesh.mesh_index,
                .lod_level = 0,
        });
    }
}
