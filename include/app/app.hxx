#pragma once

#include <span>
#include <tl/expected.hpp>
#include "ArgumentParse.hxx"

#include "BindlessHeadless.hxx"
#include "BindlessSet.hxx"
#include "Compiler.hxx"
#include "Constants.hxx"
#include "DeviceThreadPool.hxx"
#include "EventSystem.hxx"
#include "Forward.hxx"
#include "FrameQuery.hxx"
#include "GlobalCommandContext.hxx"
#include "ImGuiRenderer.hxx"
#include "Mesh.hxx"
#include "Numeric.hxx"
#include "Pool.hxx"
#include "Profiler.hxx"
#include "RenderSubmission.hxx"
#include "ResizeableGraph.hxx"
#include "Swapchain.hxx"

#include "app/frame.hxx"
#include "app/icon_parser.hxx"
#include "app/listeners.hxx"
#include "app/math.hxx"
#include "app/render.hxx"
#include "app/ui.hxx"
#include "scene/Scene.hxx"
#include "ui/PerformanceGraph.hxx"

#include "framework/LogWidget.hxx"

struct InstanceWithDebug;

struct AppGpuState {
    EngineOptions *opts{nullptr};
    InstanceWithDebug *instance{nullptr};

    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    struct QueueFamilyIndices {
        u32 graphics{0};
        u32 compute{0};
        u32 transfer{0};
    } queue_family_indices{};


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

    PipelineHandle skybox_pipeline{};

    PipelineHandle ssao_pipeline{};
    PipelineHandle ssao_blur_pipeline{};

    PipelineHandle bloom_threshold_pipeline{};
    PipelineHandle bloom_downsample_pipeline{};
    PipelineHandle bloom_upsample_pipeline{};

    PipelineHandle billboard_pipeline{};

    std::array<QueryPoolHandle, frames_in_flight> compute_query_pool{};
    std::array<QueryPoolHandle, frames_in_flight> graphics_query_pool{};
    std::array<QueryPoolHandle, frames_in_flight> graphics_stats_pool{};
    std::array<QueryPoolHandle, frames_in_flight> compute_stats_pool{};

    SamplerHandle linear_repeat{};
    SamplerHandle linear_clamp{};
    SamplerHandle noise_sampler{};
    ComparisonSamplerHandle depth_compare_filter{};
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

struct MeshInstanceRange {
    u32 mesh_index;
    u32 instance_count;
    u32 base_instance;
};

struct MeshInstanceRanges {
    static auto create(u32 mesh_count, u32 instance_count) {
        std::vector<MeshInstanceRange> ranges{};
        ranges.reserve(mesh_count);
        for (u32 i = 0; i < mesh_count; i++) {
            ranges.emplace_back(i, instance_count, i * instance_count);
        }
        return ranges;
    }
};

struct InstanceData {
    glm::mat4x3 transform{}; // sizeof(glm::mat4x3) == 48 bytes
    u32 lod_level{}; // sizeof(u32) == 4 bytes
    std::array<u32, 3> padding{0}; // sizeof(std::array<u32, 3>) == 12 bytes

    static auto empty() -> InstanceData { return InstanceData{glm::identity<glm::mat4x3>(), 0}; }
};
static_assert(sizeof(InstanceData) % 16 == 0, "Unexpected padding in InstanceData");

struct PendingIcon {
    std::string name;
    IconLoadDescription desc;
    std::filesystem::path path;
};


struct AppResources {
    std::array<FrameState, frames_in_flight> frames{};

    std::unique_ptr<AssetStreamer> asset_streamer;

    std::unique_ptr<FrameUBO> frame_ubo{std::make_unique<FrameUBO>()};

    std::vector<StaticMesh> meshes{};
    std::vector<MeshInstanceRange> mesh_instance_ranges;
    Vec<SubmeshMaterialOverride> submesh_material_overrides;
    u32 flushed_instance_count{0};

    auto instance_count() const { return flushed_instance_count; }

    std::vector<PointLight> all_point_lights{};
    std::vector<PointLight> all_point_lights_zero{};
    u32 light_count{0};

    BufferHandle point_lights_base{};
    AlignedRingBuffer<PointLight> point_lights_ring{};
    AlignedRingBuffer<u32> culled_light_count{};

    static constexpr u32 mesh_count = 1;
    AlignedRingBuffer<InstanceData> instance_ring{};

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
    TextureHandle ssao_output{};
    TextureHandle ssao_blurred{};
    TextureHandle ssao_blurred_temp{};

    Holder<TextureHandle> environment_cubemap{};
    TextureHandle perlin_noise{};

    BufferHandle noise_ssao_kernel{};
    BufferHandle ssao_hemisphere_kernel{};

    Vec<TextureHandle> bloom_downsample{};
    Vec<TextureHandle> bloom_upsample{};
    TextureHandle bloom_threshold{};
    u32 bloom_mip_count{6};

    static constexpr u32 max_draws_per_frame = 100'000U;
    AlignedRingBuffer<VkDrawIndexedIndirectCommand> indirect_ring{};
    AlignedRingBuffer<VkDrawMeshTasksIndirectCommandEXT> mesh_indirect_ring{};
    AlignedRingBuffer<u32> draw_material_id_ring{};

    struct FrameDrawStream {
        FrameIndirectWriter writer{};
        auto begin_frame() -> void { writer.cursor = 0; }
    } draw_stream{};

    StringMap<TextureHandle> icons_map{};
    bool icons_loaded{false};
};

struct OutlinerState {
    entt::entity last_decomposed = entt::null;
    std::unordered_map<entt::entity, glm::vec3> euler_cache;
};
struct AppScene {
    Scene scene{};
    RenderQueue render_queue{};
    entt::entity selected_entity = entt::null;
    OutlinerState outliner_state{};
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

    [[nodiscard]] auto extent() const -> VkExtent2D {
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

struct BloomConfig {
    float threshold{0.5f};
    float knee{0.1f};
    float radius{0.003f};
    float strength{1.0f};
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

    bool capture_next_frame{false};

    VkExtent2D last_viewport_extent = {0, 0};
    PendingResize pending_resize{};

    std::unique_ptr<LogWidget> log_widget{};

    SunConfig sun_config{};
    ShadowConfig shadow_config{};
    UIValueLatch<ClusterConfig> clustering_config{};

    UIValueLatch<u32> shadow_map_resolution{2048};

    LatestBuffer<double> last_compute_res;
    LatestBuffer<double> last_graphics_res;
    LatestBuffer<GraphicsGpuStats> last_g_stats;
    LatestBuffer<ComputeGpuStats> last_c_stats;

    BloomConfig bloom_config{};

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
    AppScene &scene;
};

class BindlessApp {
public:
    auto run(EngineOptions &opts, InstanceWithDebug &instance, RenderDocContext * = nullptr)
            -> tl::expected<i32, Error>;
};
