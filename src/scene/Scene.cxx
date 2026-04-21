#include "scene/Scene.hxx"

#include "scene/Components.hxx"

namespace hierarchy {

    auto get_or_add(entt::registry &reg, entt::entity e) -> HierarchyComponent & {
        if (!reg.all_of<HierarchyComponent>(e))
            reg.emplace<HierarchyComponent>(e);
        return reg.get<HierarchyComponent>(e);
    }

    auto set_parent(entt::registry &reg, Child child, Parent new_parent) -> void {
        if (child == new_parent)
            return;

        if (auto *hc = reg.try_get<HierarchyComponent>(child)) {
            if (hc->parent != entt::null && reg.valid(hc->parent)) {
                if (auto *old_p_hc = reg.try_get<HierarchyComponent>(hc->parent)) {
                    std::erase(old_p_hc->children, child);
                }
            }
            hc->parent = new_parent;
        } else if (new_parent != entt::null) {
            reg.emplace<HierarchyComponent>(child, new_parent);
        }

        if (new_parent != entt::null && reg.valid(new_parent)) {
            auto &new_p_hc = get_or_add(reg, new_parent);
            if (std::ranges::find(new_p_hc.children, child) == new_p_hc.children.end()) {
                new_p_hc.children.push_back(child);
            }
        }
    }

    auto roots(entt::registry &reg) -> std::vector<entt::entity> {
        std::vector<entt::entity> result;
        for (auto e: reg.view<MeshComponent>()) {
            bool is_root = true;
            if (reg.all_of<HierarchyComponent>(e)) {
                const auto &hc = reg.get<HierarchyComponent>(e);
                if (hc.parent != entt::null && reg.valid(hc.parent))
                    is_root = false;
            }
            if (is_root)
                result.push_back(e);
        }
        std::ranges::sort(result, [&](entt::entity a, entt::entity b) {
            return reg.get<MeshComponent>(a).name < reg.get<MeshComponent>(b).name;
        });
        return result;
    }

    auto is_descendant_of(entt::registry &reg, entt::entity parent, entt::entity potential_child) -> bool {
        if (parent == potential_child)
            return true;

        if (auto *hc = reg.try_get<HierarchyComponent>(parent)) {
            for (auto child: hc->children) {
                if (is_descendant_of(reg, child, potential_child))
                    return true;
            }
        }
        return false;
    }

} // namespace hierarchy
