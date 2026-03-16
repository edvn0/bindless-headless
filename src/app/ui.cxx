// app/ui.hxx  –  Scene Outliner v2
#include "app/ui.hxx"

#include "BindlessHeadless.hxx"
#include "FrameQuery.hxx"
#include "Pool.hxx"
#include "RenderContext.hxx"
#include "Types.hxx"
#include "app/app.hxx"
#include "scene/Components.hxx"
#include "ui/StyleGuard.hxx"

#include <algorithm>
#include <array>
#include <glm/gtx/matrix_decompose.hpp>
#include <imgui.h>
#include <ranges>
#include <string_view>
#include <unordered_set>

#include <ImGuizmo.h>
#include <glm/gtc/type_ptr.hpp>

namespace entity_factory {

    inline auto make_empty(entt::registry &reg, const std::string &name = "Entity") -> entt::entity {
        auto e = reg.create();
        auto &mc = reg.emplace<MeshComponent>(e);
        mc.name = name;
        mc.mesh_index = 0;
        auto &tc = reg.emplace<TransformComponent>(e);
        tc.local_to_world = glm::mat4x3{glm::mat4{1.f}};
        reg.emplace<HierarchyComponent>(e);
        return e;
    }

} // namespace entity_factory

// ─── Internal UI state ───────────────────────────────────────────────────────

namespace {
    constexpr auto get_all_children = [](const auto &self, entt::registry &reg, entt::entity e,
                                         std::vector<entt::entity> &result) -> void {
        auto hc = reg.try_get<HierarchyComponent>(e);
        if (!hc)
            return;
        result.reserve(result.size() + hc->children.size());
        for (auto child: hc->children) {
            result.push_back(child);
            self(self, reg, child, result);
        }
    };

    struct RenameState {
        entt::entity target = entt::null;
        char buf[256] = {};
        bool open_next = false;
    };

    RenameState s_rename{};
    bool s_open_add_component = false;

    auto property_label(const char *label) -> void {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(label);
        ImGui::TableSetColumnIndex(1);
        ImGui::PushItemWidth(-1);
    }

    auto draw_mesh_component(MeshComponent &mesh) -> void {
        ImGui::PushID("MeshComp");
        if (ImGui::CollapsingHeader("Mesh", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::BeginTable("MeshProps", 2, ImGuiTableFlags_SizingFixedFit)) {
                ImGui::TableSetupColumn("Label", 0, 80.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                property_label("Name");
                ImGui::TextUnformatted(mesh.name.c_str());

                property_label("Index");
                ImGui::Text("%u", mesh.mesh_index);

                ImGui::EndTable();
            }
        }
        ImGui::PopID();
    }

    auto draw_transform_component(TransformComponent &transform, entt::entity entity, OutlinerState &state) -> void {
        ImGui::PushID("TransformComp");
        if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (state.last_decomposed != entity) {
                glm::vec3 translation{}, scale{}, skew{};
                glm::vec4 perspective{};
                glm::quat orientation{};
                glm::decompose(glm::mat4{transform.local_to_world}, scale, orientation, translation, skew, perspective);
                state.euler_cache[entity] = glm::degrees(glm::eulerAngles(orientation));
                state.last_decomposed = entity;
            }

            auto &euler = state.euler_cache[entity];
            glm::vec3 t{}, s{}, skew{};
            glm::vec4 p{};
            glm::quat q{};
            glm::decompose(glm::mat4{transform.local_to_world}, s, q, t, skew, p);

            bool dirty = false;
            if (ImGui::BeginTable("TransformProps", 2, ImGuiTableFlags_SizingFixedFit)) {
                ImGui::TableSetupColumn("Label", 0, 80.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                property_label("Position");
                dirty |= ImGui::DragFloat3("##T", &t.x, 0.1f);

                property_label("Rotation");
                dirty |= ImGui::DragFloat3("##R", &euler.x, 0.5f, -180.f, 180.f, "%.1f°");

                property_label("Scale");
                dirty |= ImGui::DragFloat3("##S", &s.x, 0.01f, 0.001f, 100.f);

                ImGui::EndTable();
            }

            if (dirty) {
                transform.local_to_world =
                        glm::mat4x3{glm::translate(glm::mat4{1.f}, t) * glm::mat4_cast(glm::quat{glm::radians(euler)}) *
                                    glm::scale(glm::mat4{1.f}, s)};
            }
        }
        ImGui::PopID();
    }

    auto draw_add_component_popup(entt::registry &reg, entt::entity selected) -> void {
        if (s_open_add_component) {
            ImGui::OpenPopup("##add_component");
            s_open_add_component = false;
        }

        if (ImGui::BeginPopup("##add_component")) {
            ImGui::TextDisabled("Add Component");
            ImGui::Separator();

            const bool has_transform = reg.all_of<TransformComponent>(selected);
            if (has_transform)
                ImGui::TextDisabled("Transform (exists)");
            else if (ImGui::MenuItem("Transform")) {
                auto &tc = reg.emplace<TransformComponent>(selected);
                tc.local_to_world = glm::mat4x3{glm::mat4{1.f}};
            }

            const bool has_mesh = reg.all_of<MeshComponent>(selected);
            if (has_mesh)
                ImGui::TextDisabled("Mesh (exists)");
            else if (ImGui::MenuItem("Mesh")) {
                auto &mc = reg.emplace<MeshComponent>(selected);
                mc.name = "New Mesh";
                mc.mesh_index = 0;
            }

            const bool has_hierarchy = reg.all_of<HierarchyComponent>(selected);
            if (has_hierarchy)
                ImGui::TextDisabled("Hierarchy (exists)");
            else if (ImGui::MenuItem("Hierarchy"))
                reg.emplace<HierarchyComponent>(selected);

            ImGui::EndPopup();
        }
    }

    auto draw_inspector_widget(entt::registry &reg, entt::entity &selected, OutlinerState &state) -> void {
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize;
        ImGui::Begin("Inspector", nullptr, flags);

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6, 6));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 4));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 2));

        if (selected == entt::null || !reg.valid(selected)) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
            ImGui::TextWrapped("Select an entity to view properties.");
            ImGui::PopStyleColor();
            ImGui::PopStyleVar(3);
            ImGui::End();
            return;
        }

        if (auto *mesh = reg.try_get<MeshComponent>(selected))
            ImGui::TextColored(ImVec4(0.9f, 0.75f, 0.3f, 1.f), "%s", mesh->name.c_str());
        else
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.f), "<entity #%u>",
                               static_cast<u32>(entt::to_integral(selected)));

        ImGui::SameLine();

        constexpr float btn_w = 120.f;
        ImGui::SetCursorPosX(ImGui::GetContentRegionMax().x - btn_w);
        if (ImGui::Button("+ Add Component", ImVec2(btn_w, 0)))
            s_open_add_component = true;

        draw_add_component_popup(reg, selected);
        ImGui::Separator();

        if (auto *mesh = reg.try_get<MeshComponent>(selected))
            draw_mesh_component(*mesh);

        if (auto *transform = reg.try_get<TransformComponent>(selected))
            draw_transform_component(*transform, selected, state);

        if (auto *hc = reg.try_get<HierarchyComponent>(selected)) {
            ImGui::PushID("HierarchyComp");
            if (ImGui::CollapsingHeader("Hierarchy")) {
                if (hc->parent != entt::null && reg.valid(hc->parent)) {
                    const char *pname = "<unknown>";
                    if (auto *pm = reg.try_get<MeshComponent>(hc->parent))
                        pname = pm->name.c_str();
                    ImGui::Text("Parent: %s", pname);
                } else {
                    ImGui::TextDisabled("Parent: (none / root)");
                }
                ImGui::Text("Children: %zu", hc->children.size());
            }
            ImGui::PopID();
        }

        const float footer_h = ImGui::GetFrameHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y;
        if (ImGui::GetContentRegionAvail().y > footer_h)
            ImGui::SetCursorPosY(ImGui::GetWindowHeight() - footer_h);

        ImGui::Separator();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.55f, 0.15f, 0.15f, 1.f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.75f, 0.20f, 0.20f, 1.f));
        if (ImGui::Button("Destroy Entity", ImVec2(-1, 0))) {
            ImGui::OpenPopup("Destroy Entity?");
        }

        if (ImGui::BeginPopupModal("Destroy Entity?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Destroy this entity and all its children?");
            ImGui::Separator();


            if (ImGui::Button("Confirm", ImVec2(120, 0))) {
                std::vector<entt::entity> result{};
                get_all_children(get_all_children, reg, selected, result);
                std::ranges::for_each(result, [&](entt::entity e) { reg.destroy(e); });
                reg.destroy(selected);
                selected = entt::null;
                ImGui::CloseCurrentPopup();
            }

            ImGui::SameLine();

            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }

        ImGui::PopStyleColor(2);

        ImGui::PopStyleVar(3);
        ImGui::End();
    }

    auto handle_rename_popup(entt::registry &reg) -> void {
        if (s_rename.open_next) {
            ImGui::OpenPopup("##rename_entity");
            s_rename.open_next = false;
        }

        ImGui::SetNextWindowSize(ImVec2(260, 0), ImGuiCond_Always);
        if (ImGui::BeginPopupModal("##rename_entity", nullptr,
                                   ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar)) {
            ImGui::TextUnformatted("Rename entity:");
            ImGui::SetNextItemWidth(-1);
            if (ImGui::IsWindowAppearing())
                ImGui::SetKeyboardFocusHere();

            const bool confirmed = ImGui::InputText("##rename_buf", s_rename.buf, sizeof(s_rename.buf),
                                                    ImGuiInputTextFlags_EnterReturnsTrue);

            if ((confirmed || ImGui::Button("OK")) && s_rename.target != entt::null && reg.valid(s_rename.target)) {
                if (auto *mc = reg.try_get<MeshComponent>(s_rename.target))
                    mc->name = s_rename.buf;
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel"))
                ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }
    }

    auto draw_entity_node(entt::registry &reg, entt::entity entity, entt::entity &selected,
                          std::unordered_set<entt::entity> &visited) -> void {
        if (!reg.valid(entity) || visited.count(entity))
            return;
        visited.insert(entity);

        const auto *mesh = reg.try_get<MeshComponent>(entity);
        const char *label = mesh ? mesh->name.c_str() : "<entity>";
        const auto *hc = reg.try_get<HierarchyComponent>(entity);
        const bool has_children = hc && !hc->children.empty();

        ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick |
                                        ImGuiTreeNodeFlags_SpanFullWidth;

        if (selected == entity)
            node_flags |= ImGuiTreeNodeFlags_Selected;
        if (!has_children)
            node_flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;

        ImGui::PushID(static_cast<int>(entt::to_integral(entity)));

        const bool node_open = ImGui::TreeNodeEx(label, node_flags);

        if (ImGui::IsItemClicked(ImGuiMouseButton_Left) && !ImGui::IsItemToggledOpen())
            selected = entity;

        if (ImGui::BeginDragDropSource()) {
            ImGui::SetDragDropPayload("ENTITY_DND", &entity, sizeof(entity));
            ImGui::Text("Move: %s", label);
            ImGui::EndDragDropSource();
        }

        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("ENTITY_DND")) {
                const auto dragged = *static_cast<const entt::entity *>(payload->Data);
                if (dragged != entity)
                    hierarchy::set_parent(reg, dragged, entity);
            }
            ImGui::EndDragDropTarget();
        }

        if (ImGui::BeginPopupContextItem("##entity_ctx")) {
            if (ImGui::MenuItem("Rename")) {
                s_rename.target = entity;
                if (mesh)
                    std::snprintf(s_rename.buf, sizeof(s_rename.buf), "%s", mesh->name.c_str());
                else
                    s_rename.buf[0] = '\0';
                s_rename.open_next = true;
            }

            if (ImGui::MenuItem("Create Child")) {
                const std::string child_name = std::string(label) + "_child";
                const auto child = entity_factory::make_empty(reg, child_name);
                hierarchy::set_parent(reg, child, entity);
                selected = child;
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Detach from Parent"))
                hierarchy::set_parent(reg, entity, entt::null);

            ImGui::Separator();

            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 0.4f, 0.4f, 1.f));
            if (ImGui::MenuItem("Destroy")) {
                if (selected == entity)
                    selected = entt::null;
                std::vector<entt::entity> result{};
                get_all_children(get_all_children, reg, entity, result);
                std::ranges::for_each(result, [&](entt::entity e) { reg.destroy(e); });
                reg.destroy(entity);

                if (node_open && has_children)
                    ImGui::TreePop();
                ImGui::PopStyleColor();
                ImGui::EndPopup();
                ImGui::PopID();
                return;
            }
            ImGui::PopStyleColor();
            ImGui::EndPopup();
        }

        if (node_open && has_children) {
            auto sorted_children = hc->children;
            std::ranges::sort(sorted_children, [&](entt::entity a, entt::entity b) {
                auto *ma = reg.try_get<MeshComponent>(a);
                auto *mb = reg.try_get<MeshComponent>(b);
                const std::string_view na = ma ? ma->name : "";
                const std::string_view nb = mb ? mb->name : "";
                return na < nb;
            });
            for (auto child: sorted_children)
                draw_entity_node(reg, child, selected, visited);
            ImGui::TreePop();
        }

        ImGui::PopID();
    }

    auto draw_entity_tree(entt::registry &reg, entt::entity &selected) -> void {
        if (ImGui::Button("+ New Entity", ImVec2(-1, 0)))
            selected = entity_factory::make_empty(reg, "Entity");

        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("ENTITY_DND")) {
                const auto dragged = *static_cast<const entt::entity *>(payload->Data);
                hierarchy::set_parent(reg, dragged, entt::null);
            }
            ImGui::EndDragDropTarget();
        }

        ImGui::Separator();
        ImGui::BeginChild("##entity_tree", ImVec2(0, 0), false);

        std::unordered_set<entt::entity> visited;
        for (auto root: hierarchy::roots(reg))
            draw_entity_node(reg, root, selected, visited);

        ImGui::InvisibleButton("##tree_void", ImVec2(-1, std::max(10.f, ImGui::GetContentRegionAvail().y)));
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("ENTITY_DND")) {
                const auto dragged = *static_cast<const entt::entity *>(payload->Data);
                hierarchy::set_parent(reg, dragged, entt::null);
            }
            ImGui::EndDragDropTarget();
        }

        handle_rename_popup(reg);
        ImGui::EndChild();
    }

    auto draw_scene_outliner_widget(entt::registry &reg, entt::entity &selected) -> void {
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize;
        ImGui::Begin("Scene Outliner", nullptr, flags);

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6, 6));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 4));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 2));

        draw_entity_tree(reg, selected);

        ImGui::PopStyleVar(3);
        ImGui::End();
    }

} // namespace

static constexpr auto widget = [](const std::string_view name, auto &&func) {
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize;
    ImGui::Begin(name.data(), nullptr, flags);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6, 6));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 4));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 2));
    func();
    ImGui::PopStyleVar(3);
    ImGui::End();
};

auto draw_ui(AppContext &ctx, AppState &output) -> void {
    ImGuiViewport *main_vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(main_vp->WorkPos);
    ImGui::SetNextWindowSize(main_vp->WorkSize);
    ImGui::SetNextWindowViewport(main_vp->ID);

    ImGuiWindowFlags dock_flags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
                                  ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                                  ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
    ImGui::Begin("DockSpaceWindow", nullptr, dock_flags);
    ImGui::PopStyleVar(5);

    ImGuiID dockspace_id = ImGui::GetID("MainDockSpace");
    ImGui::DockSpace(dockspace_id, ImVec2(0, 0),
                     ImGuiDockNodeFlags_NoWindowMenuButton | ImGuiDockNodeFlags_NoCloseButton);
    ImGui::End();

    {
        StyleGuard vg(std::pair{ImGuiStyleVar_WindowPadding, ImVec2(0, 0)},
                      std::pair{ImGuiStyleVar_ItemSpacing, ImVec2(0, 0)},
                      std::pair{ImGuiStyleVar_FramePadding, ImVec2(0, 0)},
                      std::pair{ImGuiStyleVar_CellPadding, ImVec2(0, 0)});

        ImGui::Begin("Viewport", nullptr,0/*
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoScrollWithMouse*/);
        output.viewport_input = {};

        const auto view = ctx.res.frame_ubo->view;
        const auto &proj = ctx.res.frame_ubo->projection;

        ImVec2 avail = ImGui::GetContentRegionAvail();
        if (avail.x >= 1.0f && avail.y >= 1.0f) {
            const ImVec2 p0 = ImGui::GetCursorScreenPos();
            const ImVec2 p1 = {p0.x + avail.x, p0.y + avail.y};

            ImGui::GetWindowDrawList()->AddImage(ImTextureID{ctx.res.tonemapped.index()}, p0, p1);

            if (ctx.scene.selected_entity != entt::null && ctx.scene.scene.registry.valid(ctx.scene.selected_entity)) {
                if (auto *transform = ctx.scene.scene.registry.try_get<TransformComponent>(ctx.scene.selected_entity)) {
                    ImGuizmo::SetDrawlist(ImGui::GetWindowDrawList());
                    ImGuizmo::SetOrthographic(false);
                    ImGuizmo::SetRect(p0.x, p0.y, avail.x, avail.y);

                    auto model = glm::mat4{transform->local_to_world};

                    ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(proj), ImGuizmo::TRANSLATE,
                                         ImGuizmo::LOCAL, glm::value_ptr(model));

                    if (ImGuizmo::IsUsing()) {
                        transform->local_to_world = glm::mat4x3{model};
                        ctx.scene.outliner_state.last_decomposed = entt::null;
                    }
                }
            }

            output.viewport_input.min = p0;
            output.viewport_input.max = p1;
            output.viewport_input.focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
            output.viewport_input.hovered = ImGui::IsWindowHovered();

            const bool gizmo_active = ImGuizmo::IsOver() || ImGuizmo::IsUsing();
            output.viewport_input.imgui_blocks_mouse = gizmo_active;
            output.viewport_input.imgui_blocks_keyboard = ImGuizmo::IsUsing();
        }
    }
    ImGui::End();

    draw_scene_outliner_widget(ctx.scene.scene.registry, ctx.scene.selected_entity);
    draw_inspector_widget(ctx.scene.scene.registry, ctx.scene.selected_entity, ctx.scene.outliner_state);

    const auto &compute_res = ctx.ui.last_compute_res;
    const auto &c_stats = ctx.ui.last_c_stats;
    const auto &graphics_res = ctx.ui.last_graphics_res;
    const auto &g_stats = ctx.ui.last_g_stats;

    auto index = 0;
    if (compute_res.has_value())
        for (usize i = 0; i < compute_stages.size(); ++i)
            ctx.ui.gpu_frame_graph.push_sample(static_cast<int>(index++),
                                               compute_res[static_cast<u32>(compute_stages[i])]);
    if (graphics_res.has_value())
        for (usize i = 0; i < graphics_stages.size(); ++i)
            ctx.ui.gpu_frame_graph.push_sample(static_cast<int>(index++),
                                               graphics_res[static_cast<u32>(graphics_stages[i])]);

    widget("Performance Graphs", [&] {
        static int view_mode = 0;
        static bool shared_scale = false;

        ImGui::RadioButton("Combined View", &view_mode, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Split View", &view_mode, 1);
        if (view_mode == 1) {
            ImGui::SameLine();
            ImGui::Checkbox("Shared Scale", &shared_scale);
        }
        ImGui::Separator();

        StyleGuard g(std::pair{ImGuiStyleVar_WindowPadding, ImVec2(0, 0)},
                     std::pair{ImGuiStyleVar_FramePadding, ImVec2(0, 0)});
        if (view_mode == 0)
            ctx.ui.gpu_frame_graph.render("GPU Frame Times", ImVec2(-1, 200));
        else
            ctx.ui.gpu_frame_graph.render_split("GPU", ImVec2(-1, 80), shared_scale);
    });

    widget("Sun & Shadow Settings", [&] {
        if (ImGui::CollapsingHeader("Sun Direction", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Elevation", &ctx.ui.sun_config.elevation_degrees, 0.0f, 90.0f, "%.1f°");
            ImGui::SliderFloat("Azimuth", &ctx.ui.sun_config.azimuth_degrees, 0.0f, 360.0f, "%.1f°");
            ImGui::SliderFloat("Intensity", &ctx.ui.sun_config.intensity, 0.0f, 5.0f, "%.2f");
            if (ImGui::Button("Morning (East)")) {
                ctx.ui.sun_config.elevation_degrees = 30.f;
                ctx.ui.sun_config.azimuth_degrees = 90.f;
            }
            ImGui::SameLine();
            if (ImGui::Button("Noon (Overhead)")) {
                ctx.ui.sun_config.elevation_degrees = 80.f;
                ctx.ui.sun_config.azimuth_degrees = 0.f;
            }
            ImGui::SameLine();
            if (ImGui::Button("Sunset (West)")) {
                ctx.ui.sun_config.elevation_degrees = 10.f;
                ctx.ui.sun_config.azimuth_degrees = 270.f;
            }
        }
        if (ImGui::CollapsingHeader("Shadow Settings")) {
            ImGui::Columns(2, "ShadowSplit", true);
            // ImGui::SliderFloat("Shadow Distance", &ctx.ui.shadow_config.shadow_distance, -20000.f, 20000.f);
            // ImGui::SliderFloat("Ortho Size", &ctx.ui.shadow_config.ortho_size, 5.f, 10000.f);
            // ImGui::SliderFloat("Near Plane", &ctx.ui.shadow_config.near_plane, -50000.f, 50000.f);
            // ImGui::SliderFloat("Far Plane", &ctx.ui.shadow_config.far_plane, -50000.f, 50000.f);
            // ImGui::DragFloat3("Light Target", &ctx.ui.shadow_config.light_target.x, 0.1f);
            ImGui::DragFloat("Depth bias constant factor", &ctx.ui.shadow_config.depth_bias_constant_factor);
            ImGui::DragFloat("Depth bias clamp", &ctx.ui.shadow_config.depth_bias_clamp);
            ImGui::DragFloat("Depth bias slope factor", &ctx.ui.shadow_config.depth_bias_slope_factor);
            ImGui::NextColumn();
            const float image_size = ImGui::GetContentRegionAvail().x;
            ImGui::Text("Shadow Map Preview:");
            ImGui::ImageButton("Shadow map",
                               ImTextureRef{
                                       ctx.res.directional_shadow_map_depth.index(),
                               },
                               {
                                       image_size,
                                       image_size,
                               });
            ImGui::Columns(1);
        }
    });

    static u64 total_frame_counter = 0;
    widget("Frame Profile", [&] {
        ImGui::Text("Frame Profile [#%lu]", total_frame_counter++);
        ImGui::Separator();

        if (compute_res.has_value() && ImGui::CollapsingHeader("Compute Phases", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::BeginTable("ComputeTable", 2,
                                  ImGuiTableFlags_BordersInner | ImGuiTableFlags_RowBg |
                                          ImGuiTableFlags_SizingFixedFit)) {
                ImGui::TableSetupColumn("Phase");
                ImGui::TableSetupColumn("Time (ms)");
                ImGui::TableHeadersRow();
                const auto &t = compute_res;
                auto row_c = [&](const char *name, ComputeIndex idx) {
                    const u32 i = static_cast<u32>(idx);
                    if (i >= t.size())
                        return;
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(name);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.4f", t[i]);
                    if (c_stats.has_value() && i < c_stats.size()) {
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Indent();
                        ImGui::Text("Invocations:");
                        ImGui::Unindent();
                        ImGui::TableNextColumn();
                        ImGui::Text("%lu", (c_stats)[i].compute_shader_invocations);
                    }
                };
                row_c("Rotate geometry", ComputeIndex::RotateGeometry);
                row_c("Rotate lights", ComputeIndex::RotateLights);
                row_c("Light Clustering", ComputeIndex::LightClustering);
                row_c("SSAO", ComputeIndex::Ssao);
                row_c("SSAO Blur", ComputeIndex::SsaoBlur);
                row_c("Bloom", ComputeIndex::Bloom);
                ImGui::EndTable();
            }
        }

        if (graphics_res.has_value()) {
            ImGui::Separator();
            if (ImGui::CollapsingHeader("Graphics Phases", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::BeginTable("GraphicsTable", 2,
                                      ImGuiTableFlags_BordersInner | ImGuiTableFlags_RowBg |
                                              ImGuiTableFlags_SizingFixedFit)) {
                    ImGui::TableSetupColumn("Phase");
                    ImGui::TableSetupColumn("Time (ms)");
                    ImGui::TableHeadersRow();
                    const auto &t = graphics_res;
                    auto row_g = [&](const char *name, GraphicsIndex idx) {
                        const auto i = static_cast<u32>(idx);
                        if (i >= t.size())
                            return;
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted(name);
                        ImGui::TableNextColumn();
                        ImGui::Text("%.4f", t[i]);
                    };
                    row_g("Pre-Depth", GraphicsIndex::PreDepth);
                    row_g("GBuffer", GraphicsIndex::GBuffer);
                    row_g("Deferred", GraphicsIndex::Deferred);
                    row_g("Tonemap", GraphicsIndex::Tonemap);
                    row_g("Present", GraphicsIndex::Present);
                    row_g("ShadowMap", GraphicsIndex::ShadowMap);
                    row_g("Billboard", GraphicsIndex::Billboard);
                    ImGui::EndTable();
                }
                if (g_stats.has_value()) {
                    ImGui::Separator();
                    ImGui::Text("Geometry Totals");
                    const auto &gb = g_stats[static_cast<u32>(GraphicsIndex::GBuffer)];
                    ImGui::BulletText("Vertices: %lu", gb.input_assembly_vertices);
                    ImGui::BulletText("Primitives: %lu", gb.input_assembly_primitives);
                    ImGui::BulletText("Fragment Invocations: %lu", gb.fragment_shader_invocations);
                }
            }
        }

        if (compute_res.has_value() && graphics_res.has_value()) {
            double total_ms = 0.0;
            for (const auto &m: compute_res)
                total_ms += m;
            for (const auto &m: graphics_res)
                total_ms += m;
            const double c_ms = compute_res[static_cast<u32>(ComputeIndex::LightClustering)];
            const double c_pct = total_ms > 0.0 ? (c_ms / total_ms) * 100.0 : 0.0;
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.9f, 0.8f, 0.3f, 1.f), "Clustering is %.1f%% of GPU frame time", c_pct);
        }
    });

    widget("Render settings", [&] {
        const auto &dbg = ctx.ui.debug_mode;
        if (ImGui::BeginCombo("Cluster Debug Mode", std::format("{}", static_cast<u32>(dbg)).c_str(),
                              ImGuiComboFlags_HeightLarge)) {
            for (int i = 0; i < static_cast<int>(AppUI::ClusterDebugMode::Count); i++) {
                const auto mode = static_cast<AppUI::ClusterDebugMode>(i);
                const char *n = nullptr;
                switch (mode) {
                    using enum AppUI::ClusterDebugMode;
                    case None:
                        n = "None";
                        break;
                    case ClusterGrid:
                        n = "Cluster Grid";
                        break;
                    case LightCount:
                        n = "Light Count";
                        break;
                    case LightDensity:
                        n = "Light Density";
                        break;
                    case ClusterIndex:
                        n = "Cluster Index";
                        break;
                    case DepthSlices:
                        n = "Depth Slices";
                        break;
                    case LightHeatmap:
                        n = "Light Heatmap";
                        break;
                    case FirstLight:
                        n = "First Light";
                        break;
                    case ClusterOccupancy:
                        n = "Cluster Occupancy";
                        break;
                    default:
                        continue;
                }
                if (ImGui::Selectable(n, ctx.ui.debug_mode == mode))
                    ctx.ui.debug_mode = mode;
            }
            ImGui::EndCombo();
        }

        const auto &pv = ctx.ui.shadow_map_resolution.peek();
        if (ImGui::BeginCombo("Shadow Map Resolution", std::format("{}x{}", pv, pv).c_str(),
                              ImGuiComboFlags_HeightLarge)) {
            static constexpr std::array opts = {512u, 1024u, 2048u, 4096u, 8192u};
            for (const auto &res: opts) {
                const auto lbl = std::format("{}x{}", res, res);
                if (ImGui::Selectable(lbl.c_str(), ctx.ui.shadow_map_resolution.peek() == res)) {
                    ctx.ui.shadow_map_resolution = res;
                    ctx.gpu.scene_resize_graph.trigger_resize(ResizeTrigger::ShadowMap);
                }
            }
            ImGui::EndCombo();
        }
    });

    widget("Debug clustering", [&c = ctx] {
        ImGui::ImageButton("Clustering", ImTextureRef{c.res.debug_culling.index()},
                           {ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y});
    });

    widget("Cluster Configuration", [&] {
        auto &latch = ctx.ui.clustering_config;
        static ClusterConfig pending = latch.peek();
        static bool is_dirty = false;

        if (ImGui::CollapsingHeader("Grid Dimensions", ImGuiTreeNodeFlags_DefaultOpen)) {
            is_dirty |= ImGui::DragScalar("Tiles X", ImGuiDataType_U32, &pending.tiles_x, 1.f, nullptr, nullptr, "%u");
            is_dirty |= ImGui::DragScalar("Tiles Y", ImGuiDataType_U32, &pending.tiles_y, 1.f, nullptr, nullptr, "%u");
            is_dirty |= ImGui::DragScalar("Tiles Z", ImGuiDataType_U32, &pending.tiles_z, 1.f, nullptr, nullptr, "%u");
        }
        if (ImGui::CollapsingHeader("Frustum Settings")) {
            is_dirty |= ImGui::SliderFloat("Z Near", &pending.z_near, 0.1f, 10.0f);
            is_dirty |= ImGui::SliderFloat("Z Far", &pending.z_far, 10.0f, 10000.0f);
        }

        ImGui::Separator();
        ImGui::Text("Pending Clusters: %u", pending.tiles_x * pending.tiles_y * pending.tiles_z);

        if (is_dirty) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.4f, 0.2f, 1.f));
            if (ImGui::Button("Apply Changes")) {
                latch = pending;
                is_dirty = false;
                ctx.gpu.scene_resize_graph.trigger_resize(ResizeTrigger::Clustering);
            }
            ImGui::PopStyleColor();
            ImGui::SameLine();
            if (ImGui::Button("Clear")) {
                pending = latch.peek();
                is_dirty = false;
            }
        } else {
            ImGui::BeginDisabled();
            ImGui::Button("Up to date");
            ImGui::EndDisabled();
        }
    });

    widget("Icon Browser", [&] {
        static char filter_buf[128]{};
        static float icon_size = 32.0f;
        static bool show_labels = true;

        // Toolbar
        ImGui::SetNextItemWidth(180.f);
        ImGui::InputText("##icon_filter", filter_buf, sizeof(filter_buf));
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100.f);
        ImGui::SliderFloat("Size", &icon_size, 16.f, 96.f, "%.0fpx");
        ImGui::SameLine();
        ImGui::Checkbox("Labels", &show_labels);
        ImGui::Separator();

        const std::string_view filter{filter_buf};
        const float panel_w = ImGui::GetContentRegionAvail().x;
        const int cols = std::max(1, static_cast<int>(panel_w / (icon_size + ImGui::GetStyle().ItemSpacing.x + 4.f)));

        ImGui::BeginChild("##icon_scroll", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

        // Sort names for stable display order
        std::vector<std::string_view> names;
        names.reserve(ctx.res.icons_map.size());
        for (const auto &[name, _]: ctx.res.icons_map)
            if (filter.empty() || name.find(filter) != std::string::npos)
                names.push_back(name);
        std::ranges::sort(names);

        int col = 0;
        for (const auto &name: names) {
            const auto &handle = ctx.res.icons_map.at(std::string{name});

            ImGui::PushID(name.data());

            if (col > 0 && col < cols)
                ImGui::SameLine();
            if (col >= cols)
                col = 0;

            ImGui::BeginGroup();

            const bool clicked =
                    ImGui::ImageButton("##icon_btn", ImTextureRef{handle.index()}, ImVec2{icon_size, icon_size});

            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%.*s", static_cast<int>(name.size()), name.data());

            if (clicked)
                ImGui::SetClipboardText(name.data());

            if (show_labels) {
                // Truncate label to fit under the icon
                const float max_w = icon_size + 4.f;
                ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + max_w);
                ImGui::TextUnformatted(name.data(), name.data() + std::min(name.size(), usize{12}));
                ImGui::PopTextWrapPos();
            }

            ImGui::EndGroup();
            ImGui::PopID();

            ++col;
        }

        if (names.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.f));
            ImGui::TextUnformatted(filter.empty() ? "No icons loaded." : "No icons match filter.");
            ImGui::PopStyleColor();
        }

        ImGui::EndChild();
    });
}

auto run_ui_frame(AppContext &ctx) -> UiFrameResult {
    UiFrameResult out{};
    const VkExtent2D raw = current_extent(ctx.gpu.window);
    out.window_extent = sanitize_window_extent(raw, ctx.gpu.physical_device, ctx.gpu.surface);
    if (out.window_extent.width == 0 || out.window_extent.height == 0) {
        out.minimized = true;
        return out;
    }

    ctx.ui.gui->begin_frame(ImGuiFramebuffer(out.window_extent, ctx.gpu.swapchain.format(),
                                             ctx.gpu.ctx.texture_format(ctx.res.tonemapped),
                                             ctx.gpu.swapchain.color_space()));
    static u8 warmup = frames_in_flight;
    if (warmup > 0) [[unlikely]] {
        --warmup;
    } else {
        draw_ui(ctx, ctx.ui.app_state);
    }
    ctx.ui.gui->end_frame();

    out.desired_scene_extent =
            sanitize_scene_extent(ctx.ui.app_state.viewport_input.extent(),
                                  (ctx.ui.last_viewport_extent.width == 0 || ctx.ui.last_viewport_extent.height == 0)
                                          ? VkExtent2D{ctx.gpu.opts->width, ctx.gpu.opts->height}
                                          : ctx.ui.last_viewport_extent,
                                  ctx.gpu.physical_device, ExtentBounds{.min_dim = 1, .max_dim = 4096});
    return out;
}

auto window_center(GLFWwindow *w) -> glm::vec2 {
    int ww = 0, wh = 0;
    glfwGetWindowSize(w, &ww, &wh);
    return glm::vec2{ww * 0.5f, wh * 0.5f};
}

auto begin_cursor_capture(GLFWwindow *w, AppState &app) -> void {
    double x{}, y{};
    glfwGetCursorPos(w, &x, &y);
    app.last_mouse = glm::vec2(static_cast<float>(x), static_cast<float>(y));
    if (glfwRawMouseMotionSupported())
        glfwSetInputMode(w, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    app.cursor_captured = true;
    app.mouse_inited = true;
    app.warp_in_progress = false;
}

auto end_cursor_capture(GLFWwindow *w, AppState &app) -> void {
    if (glfwRawMouseMotionSupported())
        glfwSetInputMode(w, GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
    glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    app.cursor_captured = false;
    app.mouse_inited = false;
}
