#pragma once

#include "Forward.hxx"
#include "RenderContext.hxx"
#include "Types.hxx"


struct FontChoice {
    std::filesystem::path font_path;
    f32 size{20.0F};
};


using ImGuiFramebuffer = std::tuple<VkExtent2D, VkFormat, VkColorSpaceKHR>;

class ImGuiRenderer {
public:
    ImGuiRenderer(u32, RenderContext &, GlobalCommandContext &, Compiler &, FontChoice = {});
    ~ImGuiRenderer();

    ImGuiRenderer(ImGuiRenderer &&) = delete;
    auto operator=(ImGuiRenderer &&) -> ImGuiRenderer & = delete;

    auto update_font(FontChoice) -> void;
    auto begin_frame(ImGuiFramebuffer) -> void;
    auto end_frame(VkCommandBuffer) -> void;

    auto set_should_recompile() -> void { force_recompile = true; }

private:
    Holder<PipelineHandle> pipeline{};
    Holder<SamplerHandle> sampler{};
    Holder<TextureHandle> font_texture{};
    f32 display_scale{1.0F};
    u32 frame_index{0};

    RenderContext &ctx;
    GlobalCommandContext &command_ctx;
    Compiler &compiler;

    struct DrawableData {
        Holder<BufferHandle> vertex;
        Holder<BufferHandle> index;
        u32 index_count = 0;
        u32 vertex_count = 0;
    };
    std::vector<DrawableData> drawables{};

    bool force_recompile{false};

    auto create_pipeline(ImGuiFramebuffer) -> tl::expected<CompiledPipeline, Error>;
};
