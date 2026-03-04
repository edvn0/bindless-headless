#pragma once

#include "Types.hxx"
#include "StringPool.hxx"

#include <glm/mat4x3.hpp>
#include <entt/entity/entity.hpp>

struct TransformComponent {
    glm::mat4x3 local_to_world;
};

struct MeshComponent {
    FlyString name;
    u32 mesh_index{0};
};

struct HierarchyComponent {
    entt::entity              parent   = entt::null;
    std::vector<entt::entity> children;
};

namespace Internal {
    struct MeshRenderState {
        u32 base_instance;
    };
}
