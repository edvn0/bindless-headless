#include "scene/Scene.hxx"

#include "scene/Components.hxx"

namespace hierarchy {

    auto get_or_add(entt::registry &reg, entt::entity e) -> HierarchyComponent & {
        if (!reg.all_of<HierarchyComponent>(e))
            reg.emplace<HierarchyComponent>(e);
        return reg.get<HierarchyComponent>(e);
    }

    auto set_parent(entt::registry &reg, entt::entity child, entt::entity new_parent) -> void {
        auto &[parent, children] = get_or_add(reg, child);

        if (parent != entt::null && reg.valid(parent)) {
            auto &old_ph = get_or_add(reg, parent);
            auto &cv = old_ph.children;
            std::erase(cv, child);
        }

        parent = new_parent;

        if (new_parent != entt::null && reg.valid(new_parent)) {
            if (auto &ph = get_or_add(reg, new_parent); std::ranges::find(ph.children, child) == ph.children.end())
                ph.children.push_back(child);
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
            return reg.get<MeshComponent>(a).name< reg.get<MeshComponent>(b).name;
        });
        return result;
    }

}
