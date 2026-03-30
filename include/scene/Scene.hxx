#pragma once

#include "Components.hxx"

#include <entt/entt.hpp>

#include "Forward.hxx"

struct Scene {
    entt::registry registry;
    entt::group<entt::owned_t<MeshComponent>, entt::get_t<TransformComponent>> mesh_group =
            registry.group<MeshComponent>(entt::get<TransformComponent>);
};

namespace hierarchy {
    using Parent = entt::entity;
    using Child = entt::entity;
    auto get_or_add(entt::registry &reg, entt::entity e) -> HierarchyComponent &;
    auto set_parent(entt::registry &reg, Child, Parent) -> void;
    auto roots(entt::registry &) -> std::vector<entt::entity>;
} // namespace hierarchy
