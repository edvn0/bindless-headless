#pragma once

#include "Forward.hxx"
#include "RenderContext.hxx"
#include "Types.hxx"

#include <filesystem>
#include <tuple>
#include <vector>

struct FontChoice {
    std::filesystem::path font_path;
    f32 size{20.0F};
};

using ImGuiFramebuffer = std::tuple<VkExtent2D, VkFormat, VkFormat, VkColorSpaceKHR>;

class ImGuiRenderer {
public:
    ImGuiRenderer(GLFWwindow *main_window, u32 initial_slot_count, RenderContext &, Compiler &,
                  FontChoice = {});
    ~ImGuiRenderer();

    ImGuiRenderer(ImGuiRenderer &&) = delete;
    auto operator=(ImGuiRenderer &&) -> ImGuiRenderer & = delete;

    auto update_font(FontChoice) -> void;

    auto begin_frame(ImGuiFramebuffer main_fb) -> void;

    auto render(VkCommandBuffer cmd) -> void;
    auto end_frame() -> void;

    auto set_should_recompile() -> void {
        force_recompile_primary = true;
        force_recompile_offscreen = true;
    }

private:
    struct DrawableData {
        Holder<BufferHandle> vertex;
        Holder<BufferHandle> index;
        u32 index_count{0};
        u32 vertex_count{0};
    };

    Holder<PipelineHandle> main_pipeline{};
    Holder<PipelineHandle> offscreen_target_pipeline{};
    Holder<SamplerHandle> sampler{};
    Holder<TextureHandle> font_texture{};

    RenderContext &ctx;
    Compiler &compiler;

    f32 display_scale{1.0F};

    std::vector<DrawableData> drawables{};
    u32 slots_per_frame{0}; // how many slots we budget per frame
    u32 slot_cursor{0}; // how many slots used THIS frame
    u32 frame_cursor{0}; // which frame-in-flight slot base we're on

    bool force_recompile_primary{false};
    bool force_recompile_offscreen{false};

#ifdef NDEBUG
    bool frame_was_ended{true};
#endif

private:
    auto create_pipeline(VkFormat) -> tl::expected<CompiledPipeline, Error>;

    auto render_draw_data(VkCommandBuffer cmd, ImDrawData *dd, PipelineHandle) -> void;

    auto acquire_draw_slot() -> DrawableData &;

    auto render_additional_viewports() -> void;
};
