#include "ArgumentParse.hxx"
#include "BindlessHeadless.hxx"
#include "BindlessSet.hxx"
#include "Compiler.hxx"
#include "Constants.hxx"
#include "GlobalCommandContext.hxx"
#include "ImGuiRenderer.hxx"
#include "Logger.hxx"
#include "Mesh.hxx"
#include "Pipelines.hxx"
#include "RenderContext.hxx"
#include "SceneLoader.hxx"
#include "Swapchain.hxx"
#include "Types.hxx"

#include "framework/BaseApplication.hxx"

#include <GLFW/glfw3.h>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <imgui.h>
#include <limits>
#include <string>
#include <thread>
#include <vector>

class ImageViewer final : public BaseApplication {
    using B = BaseApplication;

public:
    ImageViewer() : BaseApplication(AppInfo{.version = {0, 0, 1}, .name = "ImageViewer"}) {}
    ~ImageViewer() override = default;

protected:
    auto on_init(const Arguments &args) -> tl::expected<void, Error> override {
        std::ignore = B::on_init(args);
        return {};
    }

    auto on_frame(Timestep) -> void override {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE)) {
            set_should_exit_app(true);
            return;
        }

        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Open scene…"))
                    m_show_open_dialog = true;
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }

        if (m_show_open_dialog) {
            ImGui::SetNextWindowSize({500, 120}, ImGuiCond_Always);
            if (ImGui::Begin("Open scene", &m_show_open_dialog)) {
                ImGui::InputText("Path", m_path_buf, sizeof(m_path_buf));
                if (ImGui::Button("Load")) {
                    load(m_path_buf);
                    m_show_open_dialog = false;
                }
                ImGui::SameLine();
                if (ImGui::Button("Cancel"))
                    m_show_open_dialog = false;
            }
            ImGui::End();
        }

        ImGui::SetNextWindowSize({900, 700}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Textures")) {
            if (m_textures.empty()) {
                ImGui::TextDisabled("No scene loaded — use File > Open scene");
            } else {
                ImGui::Text("%zu texture(s)", m_textures.size());
                ImGui::Separator();

                ImGui::BeginChild("list", {200, 0}, true);
                for (u32 i = 0; i < static_cast<u32>(m_textures.size()); ++i) {
                    const bool selected = (m_selected == i);
                    if (ImGui::Selectable(m_textures[i].name.c_str(), selected))
                        m_selected = i;
                }
                ImGui::EndChild();

                ImGui::SameLine();

                ImGui::BeginChild("preview", {0, 0});
                if (m_selected < m_textures.size()) {
                    const auto &t = m_textures[m_selected];
                    ImGui::Text("%s", t.name.c_str());
                    ImGui::Text("  %u x %u  |  %u mips  |  format %u", t.width, t.height, t.levels,
                                static_cast<u32>(t.vk_format));

                    if (t.handle.index() != 0) {
                        const float avail = ImGui::GetContentRegionAvail().x;
                        const float aspect = static_cast<float>(t.width) / static_cast<float>(std::max(1u, t.height));
                        const ImVec2 size = {avail, avail / aspect};

                        auto id = ImTextureRef{ImTextureID{t.handle.index()}};

                        ImGui::Image(id, size);
                    }
                }
                ImGui::EndChild();
            }
        }
        ImGui::End();
    }

    auto on_shutdown() -> void override {
        if (m_scene) {
            destroy(*ctx, m_scene->vertex_buffer);
            destroy(*ctx, m_scene->pos_uv_buffer);
            destroy(*ctx, m_scene->index_buffer);
            destroy(*ctx, m_scene->aabb_buffer);
            m_scene.reset();
        }
        m_textures.clear();
    }

private:
    struct Entry {
        std::string name;
        u32 width{}, height{}, levels{};
        VkFormat vk_format{};
        TextureHandle handle{};
    };

    std::optional<StaticMesh> m_scene;


    auto load(const std::filesystem::path &scene_path) -> void {
        m_textures.clear();
        m_selected = 0;

        auto mesh_result = load_scene(*ctx, scene_path);
        if (!mesh_result) {
            warn("ImageViewer: failed to load '{}': {}", scene_path.string(), mesh_result.error().message);
            return;
        }
        m_scene = mesh_result ? std::optional{std::move(*mesh_result)} : std::nullopt;
        ctx->bindless_set->need_repopulate = true;
        ctx->bindless_set->repopulate_if_needed(ctx->textures, ctx->samplers, ctx->comparison_samplers);
        gui->set_should_recompile();

        ctx->textures.for_each_live_with_skip([&](const auto &handle, const OffscreenTarget &t) {
            Entry e{};

            auto &info = *t.allocation_info;
            e.name = info.pName ? std::string{info.pName} : std::format("texture_{}", handle.index());
            e.width = t.width;
            e.height = t.height;
            e.levels = t.mip_levels;
            e.vk_format = t.format;
            e.handle = handle;
            m_textures.emplace_back(std::move(e));
        });

        info("ImageViewer: loaded {} textures from '{}'", m_textures.size(), scene_path.string());
    }

    std::vector<Entry> m_textures;
    u32 m_selected{0};
    bool m_show_open_dialog{false};
    char m_path_buf[512]{};
};

REGISTER_APPLICATION(ImageViewer)
