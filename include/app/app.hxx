#pragma once

#include <tl/expected.hpp>
#include "ArgumentParse.hxx"

#include "BindlessHeadless.hxx"
#include "BindlessSet.hxx"
#include "Compiler.hxx"
#include "Constants.hxx"
#include "EventSystem.hxx"
#include "FrameQuery.hxx"
#include "GlobalCommandContext.hxx"
#include "ImGuiRenderer.hxx"
#include "Mesh.hxx"
#include "Profiler.hxx"
#include "ResizeableGraph.hxx"
#include "Swapchain.hxx"

#include "app/frame.hxx"
#include "app/listeners.hxx"
#include "app/math.hxx"
#include "app/render.hxx"
#include "ui/PerformanceGraph.hxx"

struct InstanceWithDebug;

struct AppGpuState {
    CLIOptions *opts{nullptr};
    InstanceWithDebug *instance{nullptr};

    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    u32 graphics_index{0};
    u32 compute_index{0};
    u32 transfer_index{0};

    VkDevice device{VK_NULL_HANDLE};
    VkQueue graphics_queue{VK_NULL_HANDLE};
    VkQueue compute_queue{VK_NULL_HANDLE};
    VkQueue transfer_queue{VK_NULL_HANDLE};

    EnabledFeatureSet enabled_features{};

    TracyGpuContext tracy_graphics{};
    TracyGpuContext tracy_compute{};

    GLFWwindow *window{nullptr};
    VkSurfaceKHR surface{VK_NULL_HANDLE};

    Swapchain swapchain{};

    VmaAllocator allocator{VK_NULL_HANDLE};

    ComputeTimeline tl_compute{};
    GraphicsTimeline tl_graphics{};
    TransferTimeline tl_transfer{};

    BindlessSet bindless{};

    ResizeGraph window_resize_graph{};
    ResizeGraph scene_resize_graph{};

    RenderContext ctx{};

    std::unique_ptr<Compiler> compiler{};

    VkSampleCountFlagBits msaa_samples{VK_SAMPLE_COUNT_1_BIT};
};

struct AppPipelines {
    ShaderHandle fullscreen_vs{};

    PipelineHandle flags_pipeline{};
    PipelineHandle compact_pipeline{};
    PipelineHandle finalise_compact_pipeline{};
    PipelineHandle debug_point_light_pipeline{};
    PipelineHandle debug_light_clustering{};

    PipelineHandle cube_rotation_pipeline{};
    PipelineHandle light_rotation_pipeline{};

    PipelineHandle gbuffer_pipeline_mrt{};
    PipelineHandle gbuffer_pipeline_lighting{};
    PipelineHandle predepth_pipeline{};
    PipelineHandle predepth_alpha_pipeline{};
    PipelineHandle tonemap_pipeline{};
    PipelineHandle cluster_build_groups_pipeline{};
    PipelineHandle present_pipeline{};
    PipelineHandle directional_shadow_map_pipeline{};
    PipelineHandle directional_shadow_map_alpha_pipeline{};

    std::array<QueryPoolHandle, frames_in_flight> compute_query_pool{};
    std::array<QueryPoolHandle, frames_in_flight> graphics_query_pool{};
    std::array<QueryPoolHandle, frames_in_flight> graphics_stats_pool{};
    std::array<QueryPoolHandle, frames_in_flight> compute_stats_pool{};

    SamplerHandle linear_repeat{};
    SamplerHandle linear_clamp{};
    SamplerHandle noise_sampler{};
    SamplerHandle depth_compare_filter{};
};

struct ShadowConfig {
    glm::mat4 light_view_proj{1.0f};
    glm::vec3 light_target{0.0f, 0.0f, 0.0f}; // Where shadow camera looks
    float shadow_distance = 50.0f; // How far from target
    float ortho_size = 30.0f; // Width/height of ortho frustum
    float near_plane = 0.1f;
    float far_plane = 100.0f;

    float depth_bias_constant_factor = 1.25f;
    float depth_bias_clamp = 0.0f;
    float depth_bias_slope_factor = 1.75f;
};

struct SunConfig {
    float elevation_degrees = 45.0f; // 0° = horizon, 90° = zenith
    float azimuth_degrees = 135.0f; // 0° = north, 90° = east, 180° = south, 270° = west
    float intensity = 1.5f;
};


struct Cluster {
    u32 light_offset;
    u32 light_count;
};

struct AppResources {
    std::array<FrameState, frames_in_flight> frames{};

    LoadedObj mesh{};

    std::vector<PointLight> all_point_lights{};
    std::vector<PointLight> all_point_lights_zero{};
    u32 light_count{0};

    BufferHandle point_lights_base{};
    AlignedRingBuffer<PointLight> point_lights_ring{};
    AlignedRingBuffer<u32> culled_light_count{};

    static constexpr u32 mesh_count = 1;
    AlignedRingBuffer<glm::mat4x3> transforms_ring{};
    u32 instance_count{0};

    AlignedRingBuffer<u32> flags{};
    AlignedRingBuffer<u32> prefix{};
    AlignedRingBuffer<PointLight> compact_lights{};

    ClusterConfig clustering_config{};
    u32 max_light_indices{0};

    AlignedRingBuffer<Cluster> clusters{};
    AlignedRingBuffer<u32> cluster_light_indices{};

    AlignedRingBuffer<FrameUBO> frame_ubo_ring{};

    TextureHandle gbuffer0{};
    TextureHandle gbuffer1{};
    TextureHandle gbuffer2{};
    TextureHandle debug_culling{};
    TextureHandle lit_hdr{};
    TextureHandle depth{};
    TextureHandle directional_shadow_map_depth{};
    TextureHandle tonemapped{};

    TextureHandle perlin_noise{};

    static constexpr u32 max_draws_per_frame = 100000U;
    AlignedRingBuffer<VkDrawIndexedIndirectCommand> indirect_ring{};
    AlignedRingBuffer<VkDrawMeshTasksIndirectCommandEXT> mesh_indirect_ring{};
    AlignedRingBuffer<u32> draw_material_id_ring{};

    struct FrameDrawStream {
        FrameIndirectWriter writer{};
        auto begin_frame() -> void { writer.cursor = 0; }
    } draw_stream{};
};


struct ViewportInput {
    ImVec2 min{};
    ImVec2 max{};

    bool hovered{false};
    bool focused{false};

    ImGuiID viewport_item_id{0};
    ImGuiID hovered_id{0};
    ImGuiID active_id{0};

    bool imgui_blocks_mouse{false};
    bool imgui_blocks_keyboard{false};

    auto extent() const -> VkExtent2D {
        return VkExtent2D{static_cast<u32>(std::max(1.0f, max.x - min.x)),
                          static_cast<u32>(std::max(1.0f, max.y - min.y))};
    }
};

struct AppState {
    bool resized{false};

    glm::vec2 last_mouse{0.0f, 0.0f};
    bool mouse_inited{false};
    EventSystem event_system{};
    ViewportInput viewport_input{};
    bool warp_in_progress{false};
    glm::vec2 warp_center{0.0f, 0.0f};

    bool cursor_captured{false}; // optional, but makes logic simpler

    CameraInput cam_in{};
    EditorCamera cam{};
};

struct PendingResize {
    VkExtent2D desired{0, 0};
    bool has{false};

    bool was_down{false};
    double last_change_time_s{0.0};
};

struct AppUI {
    AppState app_state{};
    std::unique_ptr<ImGuiRenderer> gui{};
    std::unique_ptr<efsw::FileWatcher, Deleter> watcher{};
    std::unordered_map<std::string, std::unique_ptr<efsw::FileWatchListener, Deleter>> listeners{};

    u64 frame_index{};
    std::chrono::high_resolution_clock::time_point last_frame_time{};
    double dt{0.0};
    double total_time{0.0};

    VkExtent2D last_viewport_extent = {0, 0};
    PendingResize pending_resize{};

    SunConfig sun_config{};
    ShadowConfig shadow_config{};

    UIValueLatch<u32> shadow_map_resolution{2048};

    enum class ClusterDebugMode : u32 {
        None = 0,
        ClusterGrid = 1,
        LightCount = 2,
        LightDensity = 3,
        ClusterIndex = 4,
        DepthSlices = 5,
        LightHeatmap = 6,
        FirstLight = 7,
        ClusterOccupancy = 8,
        Count
    };

    ClusterDebugMode debug_mode{ClusterDebugMode::None};

    // graphs
    PerformanceGraph<total_queries, 120> gpu_frame_graph{};
    bool graphs_initialized{false};
};

struct AppContext {
    AppGpuState &gpu;
    AppPipelines &pipes;
    AppResources &res;
    AppUI &ui;
};

class BindlessApp {
public:
    auto run(CLIOptions &opts, InstanceWithDebug &instance) -> tl::expected<i32, Error>;
};
