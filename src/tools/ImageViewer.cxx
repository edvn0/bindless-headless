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
#include "app/ui.hxx"

#include "framework/BaseApplication.hxx"
#include "framework/LogWidget.hxx"

#include <GLFW/glfw3.h>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <imgui.h>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "Types.hxx"


class ImageViewer final : public BaseApplication {
    using B = BaseApplication;

public:
    ImageViewer() : BaseApplication(AppInfo{.version = {0, 0, 1}, .name = "ImageViewer"}) {}
    ~ImageViewer() override = default;

protected:
    auto on_init(const Arguments &args) -> tl::expected<void, Error> override {
        std::ignore = B::on_init(args);
        log_widget = std::make_unique<LogWidget>(detail::Logger::instance().imgui_buffer());
        return {};
    }

    auto on_frame(Timestep) -> void override {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE)) {
            set_should_exit_app(true);
            return;
        }

        const auto id = ImGui::GetID("ImageViewerDockspace");
        ImGui::DockSpaceOverViewport(id, ImGui::GetMainViewport());

        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Open scene…"))
                    show_open_dialog_flag = true;
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }

        if (show_open_dialog_flag) {
            ImGui::SetNextWindowSize({500, 120}, ImGuiCond_Always);
            if (ImGui::Begin("Open scene", &show_open_dialog_flag)) {
                ImGui::InputText("Path", path_buffer.data(), path_buffer.size());
                if (ImGui::Button("Load")) {
                    load(path_buffer.data());
                    show_open_dialog_flag = false;
                }
                ImGui::SameLine();
                if (ImGui::Button("Cancel"))
                    show_open_dialog_flag = false;
            }
            ImGui::End();
        }

        ImGui::SetNextWindowSize({900, 700}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Textures")) {
            if (textures.empty()) {
                ImGui::TextDisabled("No scene loaded — use File > Open scene");
            } else {
                ImGui::Text("%zu texture(s)", textures.size());
                ImGui::Separator();

                ImGui::BeginChild("list", {200, 0}, true);
                for (u32 i = 0; i < static_cast<u32>(textures.size()); ++i) {
                    const bool selected = (current_selected == i);
                    if (ImGui::Selectable(textures[i].name.c_str(), selected))
                        current_selected = i;
                }
                ImGui::EndChild();

                ImGui::SameLine();

                ImGui::BeginChild("preview", {0, 0});
                if (current_selected < textures.size()) {
                    const auto &t = textures[current_selected];
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

        log_widget->draw("Log");
    }

    auto on_shutdown() -> void override {
        if (current_scene) {
            destroy(*ctx, current_scene->vertex_buffer);
            destroy(*ctx, current_scene->pos_uv_buffer);
            destroy(*ctx, current_scene->index_buffer);
            destroy(*ctx, current_scene->aabb_buffer);
            current_scene.reset();
        }
        textures.clear();
    }

private:
    struct Entry {
        std::string name;
        u32 width{}, height{}, levels{};
        VkFormat vk_format{};
        TextureHandle handle{};
    };

    std::optional<StaticMesh> current_scene;


    auto load(const std::filesystem::path &scene_path) -> void {
        textures.clear();
        current_selected = 0;

        auto mesh_result = load_scene(*ctx, scene_path);
        if (!mesh_result) {
            warn("ImageViewer: failed to load '{}': {}", scene_path.string(), mesh_result.error().message);
            return;
        }
        current_scene = mesh_result ? std::optional{std::move(*mesh_result)} : std::nullopt;
        ctx->bindless_set->need_repopulate = true;
        ctx->bindless_set->repopulate_if_needed(ctx->textures, ctx->samplers, ctx->comparison_samplers);
        gui->set_should_recompile();

        ctx->textures.for_each_live_with_skip(
                [&](const auto &handle, const OffscreenTarget &t) {
                    Entry e{};

                    auto &info = *t.allocation_info;
                    e.name = info.pName ? std::string{info.pName} : std::format("texture_{}", handle.index());
                    e.width = t.width;
                    e.height = t.height;
                    e.levels = t.mip_levels;
                    e.vk_format = t.format;
                    e.handle = handle;
                    textures.emplace_back(std::move(e));
                },
                3u);

        info("ImageViewer: loaded {} textures from '{}'", textures.size(), scene_path.string());
    }

    std::unique_ptr<LogWidget> log_widget;
    std::vector<Entry> textures;
    u32 current_selected{0};
    bool show_open_dialog_flag{false};
    std::array<char, 512> path_buffer{};
};

REGISTER_APPLICATION(ImageViewer)
