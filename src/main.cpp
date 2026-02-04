#include "AlignedRingBuffer.hxx"
#include "Allocator.hxx"
#include "ArgumentParse.hxx"
#include "BindlessHeadless.hxx"
#include "BindlessSet.hxx"
#include "Buffer.hxx"
#include "Camera.hxx"
#include "Compiler.hxx"
#include "CompilerGlue.hxx"
#include "GlobalCommandContext.hxx"
#include "ImageOperations.hxx"
#include "Logger.hxx"
#include "PipelineCache.hxx"
#include "Pipelines.hxx"
#include "Pool.hxx"
#include "Reflection.hxx"
#include "RenderContext.hxx"
#include "ResizeableGraph.hxx"
#include "Swapchain.hxx"


#include <GLFW/glfw3.h>
#include <cassert>
#include <chrono>
#include <deque>
#include <efsw/efsw.hpp>
#include <execution>
#include <future>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/packing.hpp>
#include <iostream>
#include <ranges>
#include <thread>

#include "3PP/PerlinNoise.hpp"
#include "3PP/stb_image.h"
#include "Profiler.hxx"

#include "Constants.hxx"
#include "Mesh.hxx"
#include "Types.hxx"
#include "vulkan/vulkan_core.h"

struct ClusterConfig {
    u32 tiles_x;
    u32 tiles_y;
    u32 tiles_z;
    u32 cluster_count;
    float z_near;
    float z_far;
    float log_z_scale;
};
auto compute_cluster_config_preset(u32 width, u32 height, float z_near, float z_far) {
    u32 tiles_x, tiles_y;
    u32 tiles_z = 16u; // standard depth slices

    // Determine grid based on resolution
    if (width <= 1280 && height <= 720) {
        // 720p and below: 10x6 grid (128x120 px/tile @ 1280x720)
        tiles_x = 10u;
        tiles_y = 6u;
    } else if (width <= 1920 && height <= 1080) {
        // 1080p: 16x9 grid (120x120 px/tile @ 1920x1080)
        tiles_x = 16u;
        tiles_y = 9u;
    } else if (width <= 2560 && height <= 1440) {
        // 1440p: 20x12 grid (128x120 px/tile @ 2560x1440)
        tiles_x = 20u;
        tiles_y = 12u;
    } else if (width <= 3840 && height <= 2160) {
        // 4K: 32x18 grid (120x120 px/tile @ 3840x2160)
        tiles_x = 32u;
        tiles_y = 18u;
    } else {
        // Higher resolutions: compute dynamically with 120px target
        tiles_x = (width + 119u) / 120u;
        tiles_y = (height + 119u) / 120u;
        tiles_x = std::min(tiles_x, 64u);
        tiles_y = std::min(tiles_y, 64u);
    }

    u32 cluster_count = tiles_x * tiles_y * tiles_z;
    float log_z_scale = static_cast<float>(tiles_z) / std::log2f(z_far / z_near);

    return ClusterConfig{
            .tiles_x = tiles_x,
            .tiles_y = tiles_y,
            .tiles_z = tiles_z,
            .cluster_count = cluster_count,
            .z_near = z_near,
            .z_far = z_far,
            .log_z_scale = log_z_scale,
    };
}
auto compute_cluster_config(u32 width, u32 height, float z_near, float z_far,
                            u32 target_tile_size = 128u, // pixels per tile (adjust for performance)
                            u32 depth_slices = 16u) // number of depth slices
{
    u32 tiles_x = (width + target_tile_size - 1u) / target_tile_size;
    u32 tiles_y = (height + target_tile_size - 1u) / target_tile_size;

    tiles_x = std::max(1u, std::min(tiles_x, 64u));
    tiles_y = std::max(1u, std::min(tiles_y, 64u));

    u32 tiles_z = depth_slices;
    u32 cluster_count = tiles_x * tiles_y * tiles_z;

    float log_z_scale = static_cast<float>(tiles_z) / std::log2f(z_far / z_near);

    return ClusterConfig{
            .tiles_x = tiles_x,
            .tiles_y = tiles_y,
            .tiles_z = tiles_z,
            .cluster_count = cluster_count,
            .z_near = z_near,
            .z_far = z_far,
            .log_z_scale = log_z_scale,
    };
}


struct PointLight {
    std::array<float, 4> position_radius;
    std::array<float, 4> colour_intensity;
};
constexpr auto spawn_lights_in_aabb = [](AABB const &aabb, std::span<PointLight> lights,
                                         std::default_random_engine &rng) -> void {
    auto x_distrib = std::uniform_real_distribution{aabb.min.x, aabb.max.x};
    auto y_distrib = std::uniform_real_distribution{aabb.min.y, aabb.max.y};
    auto z_distrib = std::uniform_real_distribution{aabb.min.z, aabb.max.z};
    auto radius_distrib = std::uniform_real_distribution{0.1F, 3.0F};
    auto intensity_distrib = std::uniform_real_distribution{100.0F, 500.0F};

    auto color_distribution = std::uniform_real_distribution{0.0F, 1.0F};

    for (size_t idx = 0; idx < lights.size(); ++idx) {
        auto &[position_radius, colour_intensity] = lights[idx];

        auto const intensity = intensity_distrib(rng);
        auto const radius = radius_distrib(rng);

        position_radius = {x_distrib(rng), y_distrib(rng), z_distrib(rng), radius};

        colour_intensity = {color_distribution(rng), color_distribution(rng), color_distribution(rng), intensity};
    }
};

auto msaa_from_cli = [](u32 v) -> VkSampleCountFlagBits {
    switch (v) {
        case 1:
            return VK_SAMPLE_COUNT_1_BIT;
        case 2:
            return VK_SAMPLE_COUNT_2_BIT;
        case 4:
            return VK_SAMPLE_COUNT_4_BIT;
        case 8:
            return VK_SAMPLE_COUNT_8_BIT;
        case 16:
            return VK_SAMPLE_COUNT_16_BIT;
        case 32:
            return VK_SAMPLE_COUNT_32_BIT;
        case 64:
            return VK_SAMPLE_COUNT_64_BIT;
        case 0:
            return VkSampleCountFlagBits{};
        default:
            return VK_SAMPLE_COUNT_1_BIT;
    }
};

template<typename Stamp>
static inline auto write_ts(VkCommandBuffer cmd, const QueryPoolState &qs, VkPipelineStageFlags2 stage, Stamp s)
        -> void {
    vkCmdWriteTimestamp2(cmd, stage, qs.pool, static_cast<u32>(s));
}

static inline auto begin_stats(VkCommandBuffer cmd, const QueryPoolState &qs, const auto query) -> void {
    vkCmdBeginQuery(cmd, qs.pool, static_cast<u32>(query), 0);
}

static inline auto end_stats(VkCommandBuffer cmd, const QueryPoolState &qs, const auto query) -> void {
    vkCmdEndQuery(cmd, qs.pool, static_cast<u32>(query));
}


constexpr auto read_timestamp_pair_ms_any = [](const auto &render_context, QueryPoolHandle h, const auto begin_idx,
                                               const auto end_idx) -> std::optional<double> {
    const auto *qs = render_context.query_pools.get(h);
    if (!qs) {
        return std::nullopt;
    }

    const u32 count = qs->query_count;
    if (static_cast<u32>(begin_idx) >= count || static_cast<u32>(end_idx) >= count) {
        return std::nullopt;
    }

    std::vector<u64> stamps(count, 0);

    const VkResult r =
            vkGetQueryPoolResults(render_context.get_device(), qs->pool, 0, count, stamps.size() * sizeof(u64),
                                  stamps.data(), sizeof(u64), VK_QUERY_RESULT_64_BIT);

    if (r == VK_NOT_READY) {
        return std::nullopt;
    }
    if (r != VK_SUCCESS) {
        return std::nullopt;
    }

    const u64 dt_ticks = stamps[static_cast<u32>(end_idx)] - stamps[static_cast<u32>(begin_idx)];
    const double dt_ns = static_cast<double>(dt_ticks) * qs->timestamp_period_ns;
    return dt_ns * 1e-6;
};

constexpr auto read_timestamp_pairs_ms = [](const auto &render_context,
                                            QueryPoolHandle h) -> std::optional<std::vector<double>> {
    const auto *qs = render_context.query_pools.get(h);
    if (!qs) {
        return std::nullopt;
    }

    const u32 count = qs->query_count;
    if (count < 2 || (count % 2) != 0) {
        return std::nullopt;
    }

    std::vector<u64> stamps(count, 0);

    const VkResult r =
            vkGetQueryPoolResults(render_context.get_device(), qs->pool, 0, count, stamps.size() * sizeof(u64),
                                  stamps.data(), sizeof(u64), VK_QUERY_RESULT_64_BIT);

    if (r == VK_NOT_READY) {
        return std::nullopt;
    }
    if (r != VK_SUCCESS) {
        return std::nullopt;
    }

    std::vector<double> out{};
    out.reserve(count / 2);

    for (u32 i = 0; i < count; i += 2) {
        const u64 dt_ticks = stamps[i + 1] - stamps[i];
        const double dt_ns = static_cast<double>(dt_ticks) * qs->timestamp_period_ns;
        out.push_back(dt_ns * 1e-6);
    }

    return out;
};

constexpr auto current_extent = [](GLFWwindow *win) {
    int fbw{0};
    int fbh{0};
    glfwGetFramebufferSize(win, &fbw, &fbh);
    return VkExtent2D{.width = static_cast<u32>(std::max(fbw, 0)), .height = static_cast<u32>(std::max(fbh, 0))};
};

auto clamp_msaa_samples = [](VkPhysicalDevice physical_device,
                             VkSampleCountFlagBits requested) -> VkSampleCountFlagBits {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical_device, &props);

    const VkSampleCountFlags supported =
            props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts;

    if (requested == VK_SAMPLE_COUNT_1_BIT) {
        return VK_SAMPLE_COUNT_1_BIT;
    }

    if ((supported & requested) != 0) {
        return requested;
    }

    if ((supported & VK_SAMPLE_COUNT_64_BIT) && requested > VK_SAMPLE_COUNT_64_BIT)
        return VK_SAMPLE_COUNT_64_BIT;
    if ((supported & VK_SAMPLE_COUNT_32_BIT) && requested >= VK_SAMPLE_COUNT_32_BIT)
        return VK_SAMPLE_COUNT_32_BIT;
    if ((supported & VK_SAMPLE_COUNT_16_BIT) && requested >= VK_SAMPLE_COUNT_16_BIT)
        return VK_SAMPLE_COUNT_16_BIT;
    if ((supported & VK_SAMPLE_COUNT_8_BIT) && requested >= VK_SAMPLE_COUNT_8_BIT)
        return VK_SAMPLE_COUNT_8_BIT;
    if ((supported & VK_SAMPLE_COUNT_4_BIT) && requested >= VK_SAMPLE_COUNT_4_BIT)
        return VK_SAMPLE_COUNT_4_BIT;
    if ((supported & VK_SAMPLE_COUNT_2_BIT) && requested >= VK_SAMPLE_COUNT_2_BIT)
        return VK_SAMPLE_COUNT_2_BIT;

    return VK_SAMPLE_COUNT_1_BIT;
};

struct GraphicsGpuStats {
    u64 input_assembly_vertices;
    u64 input_assembly_primitives;
    u64 vertex_shader_invocations;
    u64 clipping_invocations;
    u64 clipping_primitives;
    u64 fragment_shader_invocations;
    u64 mesh_shader_invocations;
    u64 task_shader_invocations;
};

struct ComputeGpuStats {
    u64 compute_shader_invocations;
};


auto read_graphics_stats = [](auto &ctx, auto &device, const auto h) -> std::optional<GraphicsGpuStats> {
    const auto *qs = ctx.query_pools.get(h);
    if (!qs)
        return std::nullopt;

    std::array<u64, 8> stats{}; // Match the number of statistics you requested
    const auto r = vkGetQueryPoolResults(device, qs->pool, 0, 1, // Query index 0, count 1
                                         sizeof(stats), stats.data(), sizeof(u64), VK_QUERY_RESULT_64_BIT);

    if (r != VK_SUCCESS)
        return std::nullopt;

    return GraphicsGpuStats{
            .input_assembly_vertices = stats[0],
            .input_assembly_primitives = stats[1],
            .vertex_shader_invocations = stats[2],
            .clipping_invocations = stats[3],
            .clipping_primitives = stats[4],
            .fragment_shader_invocations = stats[5],
            .mesh_shader_invocations = stats[7],
            .task_shader_invocations = stats[6],
    };
};

auto read_compute_stats = [](auto &ctx, auto &device, const auto h) -> std::optional<ComputeGpuStats> {
    const auto *qs = ctx.query_pools.get(h);
    if (!qs)
        return std::nullopt;

    std::array<u64, 1> stats{}; // Match the number of statistics you requested
    const auto r = vkGetQueryPoolResults(device, qs->pool, 0, 1, // Query index 0, count 1
                                         sizeof(stats), stats.data(), sizeof(u64), VK_QUERY_RESULT_64_BIT);

    if (r != VK_SUCCESS)
        return std::nullopt;

    return ComputeGpuStats{
            .compute_shader_invocations = stats[0],
    };
};

struct FrustumPlane {
    glm::vec4 plane; // xyz = normal, w = distance
};


glm::mat4 PerspectiveRH_ReverseZ_Inf(float fovYRadians, float aspect, float zNear) {
    const float f = 1.0f / tanf(fovYRadians * 0.5f);

    glm::mat4 m{0.0f};

    m[0][0] = f / aspect;
    m[1][1] = f;
    m[2][3] = -1.0f;
    m[3][2] = zNear;

    // Vulkan: Z ∈ [0, 1], reverse-Z, infinite far plane
    m[2][2] = 0.0f;

    return m;
}


auto fill_zeros(VkCommandBuffer cmd, auto &buffers_ctx, auto &&...buffer_handles) {
    (vkCmdFillBuffer(cmd, buffers_ctx.get(buffer_handles)->buffer(), 0, VK_WHOLE_SIZE, 0), ...);
}

constexpr auto viewport_scissors(VkExtent2D extent) {
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = static_cast<float>(extent.height);
    viewport.width = static_cast<float>(extent.width);
    viewport.height = -static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = VkOffset2D{0, 0};
    scissor.extent = extent;
    return std::make_pair(viewport, scissor);
}

auto extract_frustum_planes = [](const glm::mat4 &inv_proj) -> std::array<FrustumPlane, 6> {
    // 1. Correct NDC Corners for ZO (0 to 1)
    constexpr std::array<glm::vec4, 8> ndc_corners = {
            glm::vec4{-1, -1, 0, 1}, {1, -1, 0, 1}, {-1, 1, 0, 1}, {1, 1, 0, 1}, // Near (0-3)
            glm::vec4{-1, -1, 1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}, {1, 1, 1, 1} // Far  (4-7)
    };

    glm::vec3 v[8];
    for (int i = 0; i < 8; ++i) {
        glm::vec4 p = inv_proj * ndc_corners[i];
        v[i] = glm::vec3(p) / p.w;
    }

    auto compute_plane = [](glm::vec3 a, glm::vec3 b, glm::vec3 c) {
        // This order ensures the normal points INSIDE the frustum
        glm::vec3 normal = glm::normalize(glm::cross(c - a, b - a));
        return glm::vec4(normal, -glm::dot(normal, a));
    };

    std::array<FrustumPlane, 6> planes{};
    planes[0].plane = compute_plane(v[0], v[2], v[4]); // Left
    planes[1].plane = compute_plane(v[1], v[5], v[3]); // Right
    planes[2].plane = compute_plane(v[0], v[4], v[1]); // Bottom
    planes[3].plane = compute_plane(v[2], v[3], v[6]); // Top
    planes[4].plane = compute_plane(v[0], v[1], v[2]); // Near
    planes[5].plane = compute_plane(v[4], v[6], v[5]); // Far

    return planes;
};

struct FrameIndirectWriter {
    u32 cursor{0}; // in commands, not bytes

    auto allocate(u32 count) -> u32 {
        u32 base = cursor;
        cursor += count;
        return base;
    }
};

struct DrawRanges {
    u32 opaque_base;
    u32 opaque_count;
    u32 alpha_base;
    u32 alpha_count;
};

static auto write_mesh_indirect(RenderContext &ctx, u32 frame_index, FrameIndirectWriter &w,
                                AlignedRingBuffer<VkDrawIndexedIndirectCommand> &cmd_ring,
                                AlignedRingBuffer<u32> &material_id_ring, const MeshData &mesh, u32 instance_count,
                                u32 first_instance) -> DrawRanges {
    const u32 total_submeshes = static_cast<u32>(mesh.submeshes.size());
    const u32 base = w.allocate(total_submeshes);

    std::vector<VkDrawIndexedIndirectCommand> opaque_cmds, alpha_cmds;
    std::vector<u32> opaque_mats, alpha_mats;

    for (const auto &s: mesh.submeshes) {
        VkDrawIndexedIndirectCommand c{.indexCount = s.index_count,
                                       .instanceCount = instance_count,
                                       .firstIndex = s.index_offset,
                                       .vertexOffset = 0,
                                       .firstInstance = first_instance};

        if (s.alpha_tested) {
            alpha_cmds.push_back(c);
            alpha_mats.push_back(s.material_id);
        } else {
            opaque_cmds.push_back(c);
            opaque_mats.push_back(s.material_id);
        }
    }

    const u32 opaque_count = static_cast<u32>(opaque_cmds.size());
    const u32 alpha_count = static_cast<u32>(alpha_cmds.size());

    // Write opaque commands first
    if (opaque_count > 0) {
        cmd_ring.write_elements(ctx, frame_index, base, std::span(opaque_cmds));
        material_id_ring.write_elements(ctx, frame_index, base, std::span(opaque_mats));
    }

    // Write alpha commands immediately after
    if (alpha_count > 0) {
        cmd_ring.write_elements(ctx, frame_index, base + opaque_count, std::span(alpha_cmds));
        material_id_ring.write_elements(ctx, frame_index, base + opaque_count, std::span(alpha_mats));
    }

    return {base, opaque_count, base + opaque_count, alpha_count};
}


struct FrameUBO {
    glm::mat4 view;
    glm::mat4 projection;
    glm::mat4 view_projection;
    glm::mat4 inv_projection;
    glm::mat4 inv_view_projection;
    glm::vec4 camera_position;
    std::array<FrustumPlane, 6> frustum_planes; // left, right, bottom, top, near, far
    glm::vec4 sun_direction_intensity;
};


auto generate_perlin(auto w, auto h) -> std::vector<std::uint8_t, default_allocator<u8>> {
    std::vector<std::uint8_t, default_allocator<u8>> data;
    data.resize(w * h);
    const auto seed = static_cast<u32>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const siv::PerlinNoise pn{seed};

    auto z_offset = 0.0;
    for (auto y = 0; y < h; ++y) {
        const auto row_z = z_offset + static_cast<double>(y) * 0.01;
        for (auto x = 0; x < w; ++x) {
            const auto nx = static_cast<double>(x) / static_cast<double>(w);
            auto ny = static_cast<double>(y) / static_cast<double>(h);
            auto value = pn.noise3D(nx * 8.0, ny * 8.0, row_z);
            value = (value + 1.0) / 2.0;
            data[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)] =
                    static_cast<std::uint8_t>(value * 255.0);
        }
        z_offset += 0.0001;
    }

    return data;
}

static VkBool32 debug_callback(const VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                               VkDebugUtilsMessageTypeFlagsEXT,
                               const VkDebugUtilsMessengerCallbackDataEXT *callback_data, void *) {
    if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        error("Validation layer: {}", callback_data->pMessage);
    }

    return VK_FALSE;
}


class ShaderSourceCodeChangeListener final : public efsw::FileWatchListener {
    ResizeGraph *resize_graph;

    // Threading & Debounce State
    std::thread worker_thread;
    std::mutex work_mutex;
    std::condition_variable work_cv;
    std::atomic<bool> should_exit{false};

    // Map of filename to last modification time to handle multiple files
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> pending_files;
    const std::chrono::milliseconds debounce_delay{100};

public:
    ShaderSourceCodeChangeListener(ResizeGraph *r) : resize_graph(r) {
        worker_thread = std::thread(&ShaderSourceCodeChangeListener::worker_loop, this);
    }

    ~ShaderSourceCodeChangeListener() {
        should_exit = true;
        work_cv.notify_all();
        if (worker_thread.joinable())
            worker_thread.join();
    }

    void handleFileAction(efsw::WatchID, const std::string &dir, const std::string &filename, efsw::Action action,
                          std::string) override {
        if (action == efsw::Actions::Modified || action == efsw::Actions::Add) {
            std::lock_guard<std::mutex> lock(work_mutex);
            // Update the "last seen" time for this file
            pending_files[dir + filename] = std::chrono::steady_clock::now();
            work_cv.notify_one();
        }
    }

private:
    void worker_loop() {
        while (!should_exit) {
            std::string file_to_compile;

            {
                std::unique_lock<std::mutex> lock(work_mutex);

                // Wait until there is work OR we need to shut down
                work_cv.wait(lock, [this] { return !pending_files.empty() || should_exit; });

                if (should_exit)
                    return;

                auto now = std::chrono::steady_clock::now();
                auto it = pending_files.begin();

                // Check if the oldest pending file has aged enough
                if (now - it->second >= debounce_delay) {
                    file_to_compile = it->first;
                    pending_files.erase(it);
                } else {
                    // Not ready yet, sleep until it is ready
                    work_cv.wait_for(lock, debounce_delay);
                    continue;
                }
            }

            if (!file_to_compile.empty()) {
                compile_shader(file_to_compile);
            }
        }
    }

    void compile_shader(const std::string &path) {
        info("Shader changed: {}", path);
        resize_graph->trigger_resize(ResizeTrigger::Shaders);
    }
};

struct Deleter {
    template<typename T>
    auto operator()(T *t) noexcept -> void {
        delete t;
    }
};

struct AppState {
    bool resized{false};

    glm::vec2 last_mouse{0.0f, 0.0f};
    bool mouse_inited{false};

    CameraInput cam_in{};
    EditorCamera cam{};
};

static auto fill_frame_ubo_from_camera(FrameUBO &ubo, const EditorCamera &cam, VkExtent2D extent, float fov_y_radians,
                                       float z_near) -> void {
    const float aspect = static_cast<float>(extent.width) / std::max(1.0f, static_cast<float>(extent.height));

    ubo.view = cam.view_matrix();
    ubo.projection = PerspectiveRH_ReverseZ_Inf(fov_y_radians, aspect, z_near);
    ubo.inv_projection = glm::inverse(ubo.projection);
    ubo.view_projection = ubo.projection * ubo.view;
    ubo.camera_position = glm::vec4(cam.camera_position(), 1.0f);
    ubo.inv_view_projection = glm::inverse(ubo.view_projection);

    const auto normal_projection = glm::inverse(glm::perspective(fov_y_radians, aspect, 0.1F, 1000.0F));
    const auto planes = extract_frustum_planes(normal_projection);
    ubo.frustum_planes = {planes[0], planes[1], planes[2], planes[3], planes[4], planes[5]};
}

static auto write_camera_to_frame_ubo(RenderContext &ctx, AlignedRingBuffer<FrameUBO> &frame_ubo_ring, u32 frame_index,
                                      const EditorCamera &cam, VkExtent2D extent, float fov_y_radians, float z_near)
        -> void {
    FrameUBO ubo{};
    fill_frame_ubo_from_camera(ubo, cam, extent, fov_y_radians, z_near);

    frame_ubo_ring.write_field(ctx, frame_index, ubo.view, offsetof(FrameUBO, view));
    frame_ubo_ring.write_field(ctx, frame_index, ubo.projection, offsetof(FrameUBO, projection));
    frame_ubo_ring.write_field(ctx, frame_index, ubo.view_projection, offsetof(FrameUBO, view_projection));
    frame_ubo_ring.write_field(ctx, frame_index, ubo.inv_view_projection, offsetof(FrameUBO, inv_view_projection));
    frame_ubo_ring.write_field(ctx, frame_index, ubo.inv_projection, offsetof(FrameUBO, inv_projection));
    frame_ubo_ring.write_field(ctx, frame_index, ubo.camera_position, offsetof(FrameUBO, camera_position));
    frame_ubo_ring.write_field(ctx, frame_index, ubo.frustum_planes, offsetof(FrameUBO, frustum_planes));
}

auto execute(CLIOptions &opts, InstanceWithDebug &instance) -> tl::expected<int, Error> {
    auto compiler = std::make_unique<Compiler>();

    auto could_choose = pick_physical_device(instance.instance);
    if (!could_choose) {
        return tl::make_unexpected(
                Error::make_error(Error::Type::DeviceSelectionError, "Failed to choose physical device"));
    }

    auto &&[physical_device, graphics_index, compute_index, transfer_index] = *could_choose;
    auto &&[device, graphics_queue, compute_queue, transfer_queue, enabled_features] =
            create_device(physical_device, graphics_index, compute_index, transfer_index);

    TracyGpuContext tracy_graphics{};
    TracyGpuContext tracy_compute{};
    tracy_graphics.init_calibrated(instance, physical_device, device, graphics_queue, graphics_index, "Graphics Queue");
    tracy_compute.init_calibrated(instance, physical_device, device, compute_queue, compute_index, "Compute Queue");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    auto window =
            glfwCreateWindow(static_cast<i32>(opts.width), static_cast<i32>(opts.height), "Bindless", nullptr, nullptr);
    if (!window) {
        error("Could not create window");
        return 1;
    }

    VkSurfaceKHR surface{};
    vk_check(glfwCreateWindowSurface(instance.instance, window, nullptr, &surface));

    auto maybe_swapchain = Swapchain::create(SwapchainCreateInfo{.physical_device = physical_device,
                                                                 .device = device,
                                                                 .surface = surface,
                                                                 .graphics_family = graphics_index,
                                                                 .extent = VkExtent2D{opts.width, opts.height},
                                                                 .vsync = opts.vsync});
    if (!maybe_swapchain) {
        return 1;
    }

    auto swapchain = std::move(maybe_swapchain.value());

    auto pipeline_cache = std::make_unique<PipelineCache>(device, opts.pipeline_cache_dir);

    auto command_context = create_global_cmd_context(device, graphics_queue, graphics_index);

    auto allocator = create_allocator(instance.instance, physical_device, device);

    auto tl_compute = create_compute_timeline(device, compute_queue, compute_index);
    auto tl_graphics = create_graphics_timeline(device, graphics_queue, graphics_index);
    auto tl_transfer = create_transfer_timeline(device, transfer_queue, transfer_index);

    BindlessSet bindless{};
    bindless.init(device, query_bindless_caps(physical_device), 8u, 8u, 8u, 0u);
    bindless.grow_if_needed(300u, 40u, 32u, 8u);


    const VkSampleCountFlagBits requested = msaa_from_cli(opts.msaa);
    const VkSampleCountFlagBits msaa_samples = clamp_msaa_samples(physical_device, requested);
    info("MSAA requested: {}, Engine supplied: {}", static_cast<u32>(requested), static_cast<u32>(msaa_samples));

    RenderContext ctx{
            .allocator = allocator,
            .bindless_set = &bindless,
    };

    PipelineHandle flags_pipeline;
    PipelineHandle compact_pipeline;
    PipelineHandle cube_rotation_pipeline;
    PipelineHandle gbuffer_pipeline_mrt;
    PipelineHandle gbuffer_pipeline_lighting;
    PipelineHandle predepth_pipeline;
    PipelineHandle predepth_alpha_pipeline;
    PipelineHandle tonemap_pipeline;

    std::array<QueryPoolHandle, frames_in_flight> compute_query_pool{};
    std::array<QueryPoolHandle, frames_in_flight> graphics_query_pool{};
    std::array<QueryPoolHandle, frames_in_flight> graphics_stats_pool{};
    std::array<QueryPoolHandle, frames_in_flight> compute_stats_pool{};
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical_device, &props);
        const auto timestamp_period_ns = static_cast<double>(props.limits.timestampPeriod);

        const VkQueryPoolCreateFlags reset_flags =
                enabled_features.contains(VK_KHR_MAINTENANCE_9_EXTENSION_NAME) ? VK_QUERY_POOL_CREATE_RESET_BIT_KHR : 0;

        for (u32 fi = 0; fi < frames_in_flight; ++fi) {
            auto qpci = create_info<VkQueryPoolCreateInfo>();
            qpci.flags = reset_flags;
            qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qpci.queryCount = compute_query_count;

            VkQueryPool qpc = VK_NULL_HANDLE;
            vk_check(vkCreateQueryPool(device, &qpci, nullptr, &qpc));

            compute_query_pool[fi] = ctx.create_query_pool(QueryPoolState{
                    .pool = qpc, .query_count = compute_query_count, .timestamp_period_ns = timestamp_period_ns});

            set_debug_name(device, VK_OBJECT_TYPE_QUERY_POOL, qpc,
                           std::format("compute_timestamp_query_pool_frame_{}", fi));

            // --- Graphics timestamps ---
            qpci.queryCount = graphics_query_count;

            VkQueryPool qpg = VK_NULL_HANDLE;
            vk_check(vkCreateQueryPool(device, &qpci, nullptr, &qpg));

            graphics_query_pool[fi] = ctx.create_query_pool(QueryPoolState{
                    .pool = qpg, .query_count = graphics_query_count, .timestamp_period_ns = timestamp_period_ns});

            set_debug_name(device, VK_OBJECT_TYPE_QUERY_POOL, qpg,
                           std::format("graphics_timestamp_query_pool_frame_{}", fi));


            VkQueryPoolCreateInfo stats_qpci{
                    .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = reset_flags,
                    .queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS,
                    .queryCount = stats_graphics_count,
                    .pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT |
                                          VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT |
                                          VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT |
                                          VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT |
                                          VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
                                          VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT |
                                          VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT |
                                          VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT,
            };

            VkQueryPool stats_pool = VK_NULL_HANDLE;
            vk_check(vkCreateQueryPool(device, &stats_qpci, nullptr, &stats_pool));
            graphics_stats_pool[fi] = ctx.create_query_pool(QueryPoolState{
                    .pool = stats_pool,
                    .query_count = stats_graphics_count,
                    .timestamp_period_ns = 0.0, // Not used for stats
            });
            set_debug_name(device, VK_OBJECT_TYPE_QUERY_POOL, stats_pool,
                           std::format("graphics_stats_query_pool_frame_{}", fi));

            // For compute statistics
            VkQueryPoolCreateInfo compute_stats_qpci{
                    .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = reset_flags,
                    .queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS,
                    .queryCount = stats_compute_count,
                    .pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT,
            };

            VkQueryPool compute_stats = VK_NULL_HANDLE;
            vk_check(vkCreateQueryPool(device, &compute_stats_qpci, nullptr, &compute_stats));
            compute_stats_pool[fi] = ctx.create_query_pool(QueryPoolState{
                    .pool = compute_stats,
                    .query_count = stats_compute_count,
                    .timestamp_period_ns = 0.0,
            });
            set_debug_name(device, VK_OBJECT_TYPE_QUERY_POOL, compute_stats,
                           std::format("compute_stats_query_pool_frame_{}", fi));

            if (reset_flags == 0) {
                vkResetQueryPool(device, qpc, 0, compute_query_count);
                vkResetQueryPool(device, qpg, 0, graphics_query_count);
                vkResetQueryPool(device, stats_pool, 0, stats_graphics_count);
                vkResetQueryPool(device, compute_stats, 0, stats_compute_count);
            }
        }
    }
    {
        std::array<u8, 4> white{255, 255, 255, 255};
        std::array<u8, 4> black{0, 0, 0, 255};
        std::array<u8, 4> flat_normal{128, 128, 255, 255};
        auto white_handle =
                ctx.create_texture(create_image_from_span_v2(allocator, command_context, 1, 1, VK_FORMAT_R8G8B8A8_UNORM,
                                                             std::as_bytes(std::span(white)), "white-texture"));
        auto black_handle =
                ctx.create_texture(create_image_from_span_v2(allocator, command_context, 1, 1, VK_FORMAT_R8G8B8A8_UNORM,
                                                             std::as_bytes(std::span(black)), "black-texture"));
        auto flat_normal_handle = ctx.create_texture(
                create_image_from_span_v2(allocator, command_context, 1, 1, VK_FORMAT_R8G8B8A8_UNORM,
                                          std::as_bytes(std::span(flat_normal)), "flat-normals-texture"));

#ifndef NDEBUG
        assert(white_handle.index() == white_texture_index);
        assert(black_handle.index() == black_texture_index);
        assert(flat_normal_handle.index() == normal_texture_index);
#else
        (void) white_handle;
        (void) black_handle;
        (void) flat_normal_handle;
#endif
    }

    auto noise = generate_perlin(2048, 2048);
    auto perlin_handle = ctx.create_texture(create_image_from_span_v2(
            allocator, command_context, 2048u, 2048u, VK_FORMAT_R8_UNORM, std::span{noise}, "perlin_noise"));
    noise.clear();


    SamplerHandle linear_repeat;
    {
        auto ci = create_info<VkSamplerCreateInfo>();
        ci.magFilter = VK_FILTER_LINEAR;
        ci.minFilter = VK_FILTER_LINEAR;
        ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

        ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

        ci.mipLodBias = 0.0f;

        ci.maxAnisotropy = 16.0f;
        ci.anisotropyEnable = VK_TRUE;

        ci.compareEnable = VK_FALSE;
        ci.compareOp = VK_COMPARE_OP_ALWAYS;

        ci.maxLod = VK_LOD_CLAMP_NONE;
        ci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;


        linear_repeat = ctx.create_sampler(ci, "linear_repeat");
        info("Linear Repeat Sampler Index: {}", linear_repeat.index());
    }

    SamplerHandle linear_clamp_sampler_handle;
    {
        auto ci = create_info<VkSamplerCreateInfo>();
        ci.magFilter = VK_FILTER_LINEAR;
        ci.minFilter = VK_FILTER_LINEAR;
        ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, // Change;
                ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, // Change;
                ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, // Change;
                ci.compareOp = VK_COMPARE_OP_ALWAYS;
        ci.maxLod = VK_LOD_CLAMP_NONE;
        ci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

        linear_clamp_sampler_handle = ctx.create_sampler(ci, "linear_clamp");
    }
    {
        VkSamplerCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        ci.magFilter = VK_FILTER_LINEAR;
        ci.minFilter = VK_FILTER_LINEAR;
        ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        ci.minLod = 0.0f;
        ci.maxLod = VK_LOD_CLAMP_NONE;

        ci.maxAnisotropy = 16.0f;
        ci.anisotropyEnable = VK_TRUE;
        ctx.create_sampler(create_sampler(allocator, ci, "noise_sampler"));
    }

    bindless.repopulate_if_needed(ctx.textures, ctx.samplers);

    std::array<FrameState, frames_in_flight> frames{};

    TRY_PROPAGATE(cube_mesh, load_obj(ctx, command_context, "assets/meshes/Sponza-master/sponza.obj"),
                  "Failed to load cube mesh");

    auto all_point_lights = std::vector<PointLight>(opts.light_count);
    auto all_point_lights_zero = std::vector<PointLight>(opts.light_count);
    auto light_count = static_cast<u32>(all_point_lights.size());
    constexpr u32 threads_per_group = 64u;
    auto group_count = (light_count + threads_per_group - 1u) / threads_per_group;

    auto rng = std::default_random_engine{
            static_cast<u32>(std::chrono::high_resolution_clock::now().time_since_epoch().count())};

    const auto mesh_aabb = cube_mesh.mesh_aabb.scaled(0.1F);
    // Log the mesh bounds for debugging
    info("Mesh AABB: min({}, {}, {}) max({}, {}, {})", mesh_aabb.min.x, mesh_aabb.min.y, mesh_aabb.min.z,
         mesh_aabb.max.x, mesh_aabb.max.y, mesh_aabb.max.z);
    spawn_lights_in_aabb(mesh_aabb, all_point_lights, rng);

    auto point_light_handle =
            ctx.buffers.create(Buffer::from_slice<PointLight>(
                                       allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       all_point_lights, "point_light")
                                       .value());

    auto culled_light_count_handle = ctx.buffers.create(
            Buffer::from_value<u32>(allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    0u, "culled_point_light_count")
                    .value());

    constexpr auto mesh_count = 1;

    auto cubes_transform_handle = AlignedRingBuffer<glm::mat4x3>::create(ctx, mesh_count, 0u, "transforms");
    cubes_transform_handle->write_all_slots(
            ctx, glm::mat4x3(glm::scale(glm::identity<glm::mat4x4>(), glm::vec3{0.1, 0.1, 0.1})));

    auto instance_count = static_cast<u32>(mesh_count);


    std::vector zeros_lights(light_count, 0u);
    std::vector zeros_groups(group_count, 0u);
    auto flags_handle = ctx.buffers.create(
            Buffer::from_slice<u32>(allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    zeros_lights, "light_flags")
                    .value());
    auto prefix_handle = ctx.buffers.create(
            Buffer::from_slice<u32>(allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    zeros_lights, "light_prefix")
                    .value());

    auto compact_lights_handle = ctx.buffers.create(
            Buffer::zeroes(allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           sizeof(PointLight) * opts.light_count, "compact_lights")
                    .value());

    //-----------------------------------
    // Choose based on your target resolution and performance needs
    // For 1920x1080: 16x9 gives 120x120 pixel tiles
    ClusterConfig cluster_config = compute_cluster_config_preset(opts.width, opts.height, 0.1F, 1000.0F);

    // For higher resolution or more precision:
    // tiles_x = 32, tiles_y = 18, tiles_z = 24 → 13,824 clusters

    // ----------------------------------------------------------------------------
    // NEW: Clustered Culling Buffers
    // ----------------------------------------------------------------------------
    constexpr auto max_lights_per_cluster = 32u;

    // 1. Per-cluster light counts (written by Pass 1, read by Pass 2)
    std::vector<u32> zero_counts(cluster_config.cluster_count, 0u);
    auto cluster_counts_handle = ctx.buffers.create(
            Buffer::from_slice<u32>(allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    zero_counts, "cluster_counts")
                    .value());

    // 2. Per-cluster offset and count (written by Pass 2, read by Pass 3 and fragment shader)
    struct Cluster {
        u32 light_offset;
        u32 light_count;
    };

    std::vector<Cluster> zero_clusters(cluster_config.cluster_count, Cluster{0u, 0u});
    auto clusters_handle =
            ctx.buffers.create(Buffer::from_slice<Cluster>(
                                       allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       zero_clusters, "clusters")
                                       .value());

    // 3. Per-cluster write counters (written by Pass 3)
    std::vector<u32> zero_counters(cluster_config.cluster_count, 0u);
    auto cluster_counters_handle = ctx.buffers.create(
            Buffer::from_slice<u32>(allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    zero_counters, "cluster_counters")
                    .value());

    // 4. Global light index buffer
    // Size conservatively: assume avg 50 lights per cluster (tune based on profiling)
    // Or allocate max: light_count * cluster_count (wasteful but safe)
    u32 max_light_indices = cluster_config.cluster_count * max_lights_per_cluster; // Conservative estimate
    auto cluster_light_indices_handle = ctx.buffers.create(
            Buffer::zeroes(allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           sizeof(u32) * max_light_indices, "cluster_light_indices")
                    .value());

    // 5. Global counter for total light indices written
    auto global_index_count_handle = ctx.buffers.create(
            Buffer::from_value<u32>(allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    0u, "global_index_count")
                    .value());
    //------------------------------------

    auto aligned_frame_buffer_handle = AlignedRingBuffer<FrameUBO>::create(ctx, "aligned_frame_ubo_buffer").value();

    auto light_addr = ctx.device_address(point_light_handle);
    auto flags_addr = ctx.device_address(flags_handle);
    auto prefix_addr = ctx.device_address(prefix_handle);
    auto compact_addr = ctx.device_address(compact_lights_handle);
    auto culled_light_count_addr = ctx.device_address(culled_light_count_handle);

    auto stats = FrameStats{};
    FrameStats gpu_compute_ms{};
    FrameStats gpu_graphics_ms{};

    AppState app_state{};
    glfwSetWindowUserPointer(window, &app_state);
    glfwSetKeyCallback(window, [](auto w, auto k, auto, auto, auto) {
        if (k == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(w, GLFW_TRUE);
        }
    });
    glfwSetWindowSizeCallback(window, [](auto w, auto, auto) {
        auto &data = *static_cast<AppState *>(glfwGetWindowUserPointer(w));
        data.resized = true;
    });
    glfwSetFramebufferSizeCallback(window, [](auto w, auto, auto) {
        auto &data = *static_cast<AppState *>(glfwGetWindowUserPointer(w));
        data.resized = true;
    });

    glfwSetMouseButtonCallback(window, [](GLFWwindow *w, int button, int action, int) {
        auto &a = *static_cast<AppState *>(glfwGetWindowUserPointer(w));
        const bool down = (action == GLFW_PRESS);

        if (button == GLFW_MOUSE_BUTTON_LEFT)
            a.cam_in.lmb = down;
        if (button == GLFW_MOUSE_BUTTON_MIDDLE)
            a.cam_in.mmb = down;
        if (button == GLFW_MOUSE_BUTTON_RIGHT)
            a.cam_in.rmb = down;

        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            glfwSetInputMode(w, GLFW_CURSOR, down ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
            a.mouse_inited = false;
        }
    });

    glfwSetCursorPosCallback(window, [](GLFWwindow *w, double x, double y) {
        auto &a = *static_cast<AppState *>(glfwGetWindowUserPointer(w));
        const glm::vec2 p{static_cast<float>(x), static_cast<float>(y)};

        if (!a.mouse_inited) {
            a.last_mouse = p;
            a.mouse_inited = true;
            return;
        }

        const glm::vec2 d = p - a.last_mouse;
        a.last_mouse = p;

        a.cam_in.mouse_delta += d;
    });

    glfwSetScrollCallback(window, [](GLFWwindow *w, double, double yoff) {
        auto &a = *static_cast<AppState *>(glfwGetWindowUserPointer(w));
        a.cam_in.scroll_delta += static_cast<float>(yoff);
    });

    glfwShowWindow(window);
    glfwFocusWindow(window);

    TextureHandle gbuffer0_handle; // ALBEDO_AO: VK_FORMAT_R8G8B8A8_UNORM
    TextureHandle gbuffer1_handle; // Normal OCT, Roughness Metallic: VK_FORMAT_R16G16B16A16_UNORM
    TextureHandle gbuffer2_handle; // Emissive HDR: VK_FORMAT_R16G16B16A16_SFLOAT
    TextureHandle lit_hdr_handle; // VK_FORMAT_R16G16B16A16_SFLOAT
    TextureHandle depth_handle; // VK_FORMAT_D32_SFLOAT
    TextureHandle tonemapped_target_handle;

    VkExtent2D last_extent = current_extent(window);
    ResizeGraph resize_graph{};
    {
        const auto swapchain_node =
                resize_graph.add_node("swapchain", [&](VkExtent2D new_extent, const ResizeContext &) {
                    if (auto r = swapchain.recreate(new_extent); !r) {
                        vk_check(r.error());
                    }
                });

        const auto tonemapped_node =
                resize_graph.add_node("tonemapped_image", [&](VkExtent2D e, const ResizeContext &resize_context) {
                    const auto old_tonemap = tonemapped_target_handle;

                    tonemapped_target_handle = ctx.create_texture(create_offscreen_target(
                            allocator, e.width, e.height, VK_FORMAT_R8G8B8A8_SRGB, {}, "tonemapped"));
                    destroy(ctx, old_tonemap, resize_context.retire_value);
                });

        const auto offscreen_node = resize_graph.add_node("offscreen_targets", [&](VkExtent2D e,
                                                                                   const ResizeContext &rc) {
            const auto old_g0 = gbuffer0_handle;
            const auto old_g1 = gbuffer1_handle;
            const auto old_g2 = gbuffer2_handle;
            const auto old_hdr = lit_hdr_handle;
            const auto old_depth = depth_handle;

            gbuffer0_handle = ctx.create_texture(create_offscreen_target(
                    allocator, e.width, e.height, VK_FORMAT_R8G8B8A8_UNORM, {}, "gbuffer0_albedo_ao"));

            gbuffer1_handle = ctx.create_texture(create_offscreen_target(
                    allocator, e.width, e.height, VK_FORMAT_R16G16B16A16_SFLOAT, {}, "gbuffer1_normal_rough_metal"));

            gbuffer2_handle = ctx.create_texture(create_offscreen_target(
                    allocator, e.width, e.height, VK_FORMAT_R16G16B16A16_SFLOAT, {}, "gbuffer2_emissive"));

            depth_handle = ctx.create_texture(create_depth_target(allocator, e.width, e.height, VK_FORMAT_D32_SFLOAT,
                                                                  VK_SAMPLE_COUNT_1_BIT, false, "depth"));

            lit_hdr_handle = ctx.create_texture(create_offscreen_target(allocator, e.width, e.height,
                                                                        VK_FORMAT_R16G16B16A16_SFLOAT, {}, "lit_hdr"));

            destroy(ctx, old_g0, rc.retire_value);
            destroy(ctx, old_g1, rc.retire_value);
            destroy(ctx, old_g2, rc.retire_value);
            destroy(ctx, old_hdr, rc.retire_value);
            destroy(ctx, old_depth, rc.retire_value);
        });

        const auto uniforms_node = resize_graph.add_node("frame_ubo_camera", [&](VkExtent2D, const ResizeContext &) {
            // TODO: When something needs to be done with the uniforms node.
        });

#define TRY_UNWRAP(var_name, expected_expr, msg)                                                                       \
    auto var_name##_tmp = (expected_expr);                                                                             \
    if (!var_name##_tmp.has_value()) {                                                                                 \
        warn("{}: {}", msg, var_name##_tmp.error());                                                                   \
        return;                                                                                                        \
    }                                                                                                                  \
    auto var_name = std::move(var_name##_tmp.value());

        const auto pipelines_node = resize_graph.add_node(
                "pipelines",
                [&](VkExtent2D, const ResizeContext &rc) {
                    const auto old_gbuffer_pipeline_lighting = gbuffer_pipeline_lighting;
                    const auto old_cube_rotation_pipeline = cube_rotation_pipeline;
                    const auto old_gbuffer_pipeline_mrt = gbuffer_pipeline_mrt;
                    const auto old_flags_pipeline = flags_pipeline;
                    const auto old_compact_pipeline = compact_pipeline;
                    const auto old_predepth_pipeline = predepth_pipeline;
                    const auto old_predepth_alpha_pipeline = predepth_alpha_pipeline;
                    const auto old_tonemap_pipeline = tonemap_pipeline;

                    std::array<const std::string_view, 2> names = {"LightFlagsCS", "LightCompactCS"};
                    std::array<ReflectionData, names.size()> reflection_data = {};
                    TRY_UNWRAP(culling_code,
                               compiler->compile_from_file("shaders/light_cull_compact_modern.slang", std::span(names),
                                                           std::span(reflection_data)),
                               "Failed to compile light culling shader");

                    std::array<const std::string_view, 2> predepth_names{"main_vs_mdi", "fs_main"};
                    std::array<ReflectionData, predepth_names.size()> predepth_reflection{};
                    TRY_UNWRAP(predepth_code,
                               compiler->compile_from_file("shaders/predepth.slang", std::span(predepth_names),
                                                           std::span(predepth_reflection)),
                               "Failed to compile predepth shader");

                    std::array<const std::string_view, 2> tonemap_names{"vs_main", "fs_main"};
                    std::array<ReflectionData, tonemap_names.size()> tonemap_reflection{};
                    TRY_UNWRAP(tonemap_code,
                               compiler->compile_from_file("shaders/tonemap.slang", std::span(tonemap_names),
                                                           std::span(tonemap_reflection)),
                               "Failed to compile tonemap shader");

                    std::array<const std::string_view, 1> rotate_cubes_names{"rotate_cs"};
                    std::array<ReflectionData, rotate_cubes_names.size()> rotate_cubes_reflection{};
                    TRY_UNWRAP(rotate_cubes_code,
                               compiler->compile_from_file("shaders/rotate_cubes.slang", std::span(rotate_cubes_names),
                                                           std::span(rotate_cubes_reflection)),
                               "Failed to compile rotate cubes shader");


                    std::array<const std::string_view, 4> gbuffer_entry_point_names = {
                            "main_vs_mdi", "main_fs_mdi", "vs_fullscreen_main", "fs_fullscreen_main"};
                    std::array<ReflectionData, gbuffer_entry_point_names.size()> gbuffer_reflection{};
                    TRY_UNWRAP(gbuffer_mrt_and_lighting_code,
                               compiler->compile_from_file("shaders/gbuffer.slang",
                                                           std::span(gbuffer_entry_point_names),
                                                           std::span(gbuffer_reflection)),
                               "Failed to compile gbuffer shader");

                    auto &&[fp, cp] = create_compute_pipelines(device, *pipeline_cache, bindless.layout,
                                                               std::span(culling_code), std::span(names));

                    auto &&[crp] =
                            create_compute_pipelines(device, *pipeline_cache, bindless.layout,
                                                     std::span(rotate_cubes_code), std::span(rotate_cubes_names));

                    auto gbuffer_pipeline = create_gbuffer_pipeline(
                            device, *pipeline_cache, bindless.layout, gbuffer_mrt_and_lighting_code.at(0),
                            gbuffer_mrt_and_lighting_code.at(1), VK_FORMAT_R8G8B8A8_UNORM,
                            VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_D32_SFLOAT);

                    auto gbuf_light = create_deferred_lighting_graphics_pipeline(
                            device, *pipeline_cache, bindless.layout, gbuffer_mrt_and_lighting_code.at(2),
                            gbuffer_mrt_and_lighting_code.at(3), "vs_fullscreen_main", "fs_fullscreen_main",
                            VK_FORMAT_R16G16B16A16_SFLOAT);

                    auto pp = create_predepth_pipeline(device, *pipeline_cache, bindless.layout, predepth_code.at(0),
                                                       VK_FORMAT_D32_SFLOAT, msaa_samples);
                    auto pp_alpha =
                            create_predepth_pipeline(device, *pipeline_cache, bindless.layout, predepth_code.at(0),
                                                     predepth_code.at(1), VK_FORMAT_D32_SFLOAT, msaa_samples);

                    auto tp =
                            create_tonemap_pipeline(device, *pipeline_cache, bindless.layout, tonemap_code.at(0),
                                                    tonemap_code.at(1), "vs_main", "fs_main", VK_FORMAT_R8G8B8A8_SRGB);
                    gbuffer_pipeline_lighting = ctx.create_pipeline(std::move(gbuf_light));
                    cube_rotation_pipeline = ctx.create_pipeline(std::move(crp));
                    gbuffer_pipeline_mrt = ctx.create_pipeline(std::move(gbuffer_pipeline));
                    flags_pipeline = ctx.create_pipeline(std::move(fp));
                    compact_pipeline = ctx.create_pipeline(std::move(cp));
                    predepth_pipeline = ctx.create_pipeline(std::move(pp));
                    predepth_alpha_pipeline = ctx.create_pipeline(std::move(pp_alpha));
                    tonemap_pipeline = ctx.create_pipeline(std::move(tp));


                    destroy(ctx, old_gbuffer_pipeline_lighting, rc.retire_value);
                    destroy(ctx, old_cube_rotation_pipeline, rc.retire_value);
                    destroy(ctx, old_gbuffer_pipeline_mrt, rc.retire_value);
                    destroy(ctx, old_flags_pipeline, rc.retire_value);
                    destroy(ctx, old_compact_pipeline, rc.retire_value);
                    destroy(ctx, old_predepth_pipeline, rc.retire_value);
                    destroy(ctx, old_predepth_alpha_pipeline, rc.retire_value);
                    destroy(ctx, old_tonemap_pipeline, rc.retire_value);
                },
                ResizeTrigger::Shaders);


        resize_graph.add_dependency(tonemapped_node, offscreen_node);
        resize_graph.add_dependency(offscreen_node, swapchain_node);
        resize_graph.add_dependency(pipelines_node, offscreen_node);
        resize_graph.add_dependency(uniforms_node, swapchain_node);
    }
#undef TRY_UNWRAP

    resize_graph.rebuild(last_extent, ResizeContext{
                                              .ctx = ctx,
                                              .retire_value = 0,
                                      });

    constexpr u32 max_draws_per_frame = 100000U;

    auto indirect_ring = AlignedRingBuffer<VkDrawIndexedIndirectCommand>::create(
                                 ctx, max_draws_per_frame, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, "frame_indirect_cmds")
                                 .value();

    auto draw_material_id_ring =
            AlignedRingBuffer<u32>::create(ctx, max_draws_per_frame, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           "frame_draw_material_ids")
                    .value();

    struct FrameDrawStream {
        FrameIndirectWriter writer{};

        auto begin_frame() -> void { writer.cursor = 0; }
    };
    FrameDrawStream draw_stream{};

    u64 frame_index{};
    auto last_frame_time = std::chrono::high_resolution_clock::now();
    double dt = 0.0;

    std::unique_ptr<efsw::FileWatcher, Deleter> watcher(new efsw::FileWatcher(false), Deleter{});
    std::unordered_map<std::string, std::unique_ptr<efsw::FileWatchListener, Deleter>> listeners;
    listeners["update"] = std::unique_ptr<efsw::FileWatchListener, Deleter>(
            new ShaderSourceCodeChangeListener(&resize_graph), Deleter{});

    std::ignore = watcher->addWatch("shaders", listeners["update"].get(), true,
                                    {efsw::WatcherOption(efsw::Option::WinBufferSize, 128 * 1024)});
    watcher->watch();


    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        const u64 completed_now = std::min(tl_compute.completed, tl_graphics.completed);
        glfwPollEvents();

        const auto extent = current_extent(window);
        const bool window_resized = (extent.width != last_extent.width || extent.height != last_extent.height);

        ResizeTrigger manual_trigger = resize_graph.get_and_clear_triggers();

        if (window_resized || manual_trigger != ResizeTrigger::None) {
            if (extent.width == 0 || extent.height == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            last_extent = extent;

            ResizeTrigger final_trigger = manual_trigger;
            if (window_resized) {
                final_trigger = final_trigger | ResizeTrigger::Extent;
            }

            resize_graph.rebuild(extent,
                                 ResizeContext{
                                         .ctx = ctx,
                                         .retire_value = completed_now,
                                 },
                                 final_trigger);

            if (window_resized)
                continue;
        }

        auto current_frame_time = std::chrono::high_resolution_clock::now();
        dt = std::chrono::duration<double>(current_frame_time - last_frame_time).count();
        last_frame_time = current_frame_time;

        const auto frame_extent = swapchain.extent();
        auto start_time = std::chrono::high_resolution_clock::now();
        const auto bounded_frame_index = static_cast<u32>(frame_index % frames_in_flight);
        const auto last_frame_index = static_cast<u32>((frame_index + frames_in_flight - 1u) % frames_in_flight);
        draw_stream.begin_frame();
        app_state.cam.update(window, dt, app_state.cam_in);
        constexpr float fov_y = glm::radians(70.0f);
        constexpr float z_near = 0.1f;
        write_camera_to_frame_ubo(ctx, aligned_frame_buffer_handle, bounded_frame_index, app_state.cam, frame_extent,
                                  fov_y, z_near);
        {
            static double total_time = 0.0;
            total_time += dt;

            constexpr auto rads_per_second = glm::radians(20.0f);
            const float angle = static_cast<float>(total_time * rads_per_second);

            const glm::vec3 sun_dir = glm::normalize(glm::vec3(std::cos(angle), std::sin(angle), -0.4f));

            auto sun_direction_intensity = glm::vec4(sun_dir, 1.5f);
            auto offset = offsetof(FrameUBO, sun_direction_intensity);

            aligned_frame_buffer_handle.write_field(ctx, bounded_frame_index, sun_direction_intensity, offset);
        }

        const auto ranges =
                write_mesh_indirect(ctx, bounded_frame_index, draw_stream.writer, indirect_ring, draw_material_id_ring,
                                    cube_mesh.mesh, instance_count, 0u /* first_instance */
                );

        if (bindless.repopulate_if_needed(ctx.textures, ctx.samplers)) {
            resize_graph.rebuild(current_extent(window),
                                 ResizeContext{
                                         .ctx = ctx,
                                         .retire_value = completed_now,
                                 },
                                 ResizeTrigger::Shaders);
            info("Bindless set was repopulated, resizing pipelines.");
        }

        auto &fs = frames[bounded_frame_index];

        if (fs.frame_done_value > 0) {
            VkSemaphoreWaitInfo wi{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                   .pNext = nullptr,
                                   .flags = 0,
                                   .semaphoreCount = 1,
                                   .pSemaphores = &tl_graphics.timeline,
                                   .pValues = &fs.frame_done_value};
            vk_check(vkWaitSemaphores(device, &wi, UINT64_MAX));

            if (auto ms = read_timestamp_pair_ms_any(ctx, compute_query_pool[bounded_frame_index],
                                                     ComputeStamp::RotateBegin, ComputeStamp::CullEnd);
                ms.has_value()) {
                gpu_compute_ms.add_sample(ms.value());
            }
            if (auto ms = read_timestamp_pair_ms_any(ctx, graphics_query_pool[bounded_frame_index],
                                                     GraphicsStamp::PreDepthBegin, GraphicsStamp::PresentEnd);
                ms.has_value()) {
                gpu_graphics_ms.add_sample(*ms);
            }


            if (auto pipeline_stats = read_graphics_stats(ctx, device, graphics_stats_pool[bounded_frame_index]);
                pipeline_stats.has_value()) {
                volatile auto keep = *pipeline_stats;
                (void) keep;
            }
            if (auto pipeline_stats = read_compute_stats(ctx, device, compute_stats_pool[bounded_frame_index]);
                pipeline_stats.has_value()) {
                volatile auto keep = *pipeline_stats;
                (void) keep;
            }

            auto &&[a, b, c, d] = ctx.query_pools.get_multiple(
                    compute_query_pool[bounded_frame_index], graphics_query_pool[bounded_frame_index],
                    graphics_stats_pool[bounded_frame_index], compute_stats_pool[bounded_frame_index]);
            vkResetQueryPool(device, a->pool, 0, a->query_count);
            vkResetQueryPool(device, b->pool, 0, b->query_count);
            vkResetQueryPool(device, c->pool, 0, c->query_count);
            vkResetQueryPool(device, d->pool, 0, d->query_count);
        }

        auto acquired = swapchain.acquire_next_image(bounded_frame_index);
        if (!acquired) {
            const VkResult res = acquired.error();
            if (res == VK_ERROR_OUT_OF_DATE_KHR) {
                continue;
            }
            vk_check(res);
        }

        const auto swap_image_index = acquired->image_index;
        const auto frame_sync = acquired->sync;

        auto rotate_cubes_gpu_val = submit_stage(
                tl_compute, device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(tracy_compute.ctx, cmd, "RotateCubesGPU");

                    auto &&[ts, stats] = ctx.query_pools.get_multiple(compute_query_pool[bounded_frame_index],
                                                                      compute_stats_pool[bounded_frame_index]);
                    auto *pipe = ctx.pipeline_pool.get(cube_rotation_pipeline);

                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, ComputeStamp::RotateBegin);
                    begin_stats(cmd, *stats, ComputeIndex::Rotate);

                    auto *buffer = ctx.buffers.get(cubes_transform_handle->handle());

                    RotateCubesPushConstant pc{
                            .cube_count = instance_count,
                            .delta_time = static_cast<float>(dt),
                            .rads_per_second = glm::radians(20.0f),
                            .transforms = cubes_transform_handle->slot_device_address(bounded_frame_index),
                            .previous_frame_transforms = cubes_transform_handle->slot_device_address(last_frame_index),
                    };

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline);
                    vkCmdPushConstants(cmd, pipe->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                    const u32 groups = (instance_count + 63u) / 64u;
                    vkCmdDispatch(cmd, groups, 1, 1);

                    end_stats(cmd, *stats, ComputeIndex::Rotate);
                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, ComputeStamp::RotateEnd);

                    VkBufferMemoryBarrier2 mem_barrier{};
                    mem_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                    mem_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    mem_barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
                    mem_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    mem_barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
                    mem_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    mem_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    mem_barrier.buffer = buffer->buffer();
                    mem_barrier.offset =
                            static_cast<VkDeviceSize>(cubes_transform_handle->slot_offset_bytes(bounded_frame_index));
                    mem_barrier.size = static_cast<VkDeviceSize>(instance_count * sizeof(glm::mat4x3));

                    VkDependencyInfo dep_info{};
                    dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dep_info.bufferMemoryBarrierCount = 1;
                    dep_info.pBufferMemoryBarriers = &mem_barrier;

                    vkCmdPipelineBarrier2(cmd, &dep_info);
                },
                no_waits);
        fs.timeline_values[stage_index(Stage::CubeRotation)] = rotate_cubes_gpu_val;

        const std::array cube_rotate_waits{TimelineWait{
                .value = fs.timeline_values[stage_index(Stage::CubeRotation)],
                .semaphore = tl_compute.timeline,
                .stage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
        }};

        auto predepth_val = submit_stage(
                tl_graphics, device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(tracy_graphics.ctx, cmd, "Predepth");

                    auto &&[ts, stats] = ctx.query_pools.get_multiple(graphics_query_pool[bounded_frame_index],
                                                                      graphics_stats_pool[bounded_frame_index]);
                    auto &&[predepth, alpha] =
                            ctx.pipeline_pool.get_multiple(predepth_pipeline, predepth_alpha_pipeline);


                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, GraphicsStamp::PreDepthBegin);
                    begin_stats(cmd, *stats, GraphicsIndex::PreDepth);

                    auto &&depth = ctx.textures.get(depth_handle);
                    auto &&[indirect, verts, idx, materials] =
                            ctx.buffers.get_multiple(indirect_ring.handle(), cube_mesh.pos_uv_buffer,
                                                     cube_mesh.index_buffer, cube_mesh.material_buffer);

                    depth->transition_if_not_initialised(cmd, VK_IMAGE_LAYOUT_GENERAL,
                                                         {VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                                          VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT});

                    VkRenderingAttachmentInfo depth_attachment{};
                    depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    depth_attachment.imageView = depth->attachment_view;
                    depth_attachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                    depth_attachment.clearValue = {.depthStencil = {0.0f, 0}};

                    VkRenderingInfo rendering_info{};
                    rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                    rendering_info.renderArea = {.offset = {0, 0}, .extent = {frame_extent.width, frame_extent.height}};
                    rendering_info.layerCount = 1;
                    rendering_info.pDepthAttachment = &depth_attachment;

                    vkCmdBeginRendering(cmd, &rendering_info);
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, predepth->pipeline);

                    const PredepthPushConstants pc = {
                            .ubo = aligned_frame_buffer_handle.slot_device_address(bounded_frame_index),
                            .transforms = cubes_transform_handle->slot_device_address(bounded_frame_index),
                            .draw_material_ids = draw_material_id_ring.slot_device_address(bounded_frame_index),
                            .materials = materials->device_address(),
                            .base_draw_id = ranges.opaque_base,
                            .sampler_index = 0,
                    };

                    auto &&[vp, sc] = viewport_scissors(frame_extent);
                    vkCmdSetViewport(cmd, 0, 1, &vp);
                    vkCmdSetScissor(cmd, 0, 1, &sc);
                    vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_GREATER_OR_EQUAL); // Reverse-Z
                    vkCmdSetDepthBounds(cmd, 0.0F, 1.0F);
                    vkCmdSetCullMode(cmd, VK_CULL_MODE_BACK_BIT);
                    vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);

                    vkCmdBindIndexBuffer(cmd, idx->buffer(), 0, VK_INDEX_TYPE_UINT32);
                    std::array<VkBuffer, 1> buffers = {verts->buffer()};
                    std::array<VkDeviceSize, 1> offsets = {0};
                    const auto size = VkDeviceSize{verts->size()};
                    vkCmdBindVertexBuffers2(cmd, 0, 1, buffers.data(), offsets.data(), &size, nullptr);

                    // --- 1. OPAQUE PASS ---
                    if (ranges.opaque_count > 0) {
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, predepth->pipeline);

                        // Always bind descriptors AFTER binding the pipeline
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, predepth->layout, 0, 1,
                                                &bindless.set, 0, nullptr);

                        PredepthPushConstants opaque_pc = pc;
                        opaque_pc.base_draw_id = ranges.opaque_base;

                        // Opaque pre-depth typically only has a Vertex Shader
                        vkCmdPushConstants(cmd, predepth->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(opaque_pc),
                                           &opaque_pc);

                        const VkDeviceSize opaque_offset =
                                static_cast<VkDeviceSize>(indirect_ring.slot_offset_bytes(bounded_frame_index)) +
                                (ranges.opaque_base * sizeof(VkDrawIndexedIndirectCommand));

                        vkCmdDrawIndexedIndirect(cmd, indirect->buffer(), opaque_offset, ranges.opaque_count,
                                                 sizeof(VkDrawIndexedIndirectCommand));
                    }

                    // --- 2. ALPHA TESTED PASS ---
                    if (ranges.alpha_count > 0) {
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, alpha->pipeline);

                        // Re-bind descriptors if the alpha pipeline uses a different layout
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, alpha->layout, 0, 1,
                                                &bindless.set, 0, nullptr);

                        PredepthPushConstants alpha_pc = pc;
                        alpha_pc.base_draw_id = ranges.alpha_base;

                        // Alpha pre-depth MUST have Fragment access for texture sampling
                        vkCmdPushConstants(cmd, alpha->layout,
                                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                           sizeof(alpha_pc), &alpha_pc);

                        const VkDeviceSize alpha_offset =
                                static_cast<VkDeviceSize>(indirect_ring.slot_offset_bytes(bounded_frame_index)) +
                                (ranges.alpha_base * sizeof(VkDrawIndexedIndirectCommand));

                        vkCmdDrawIndexedIndirect(cmd, indirect->buffer(), alpha_offset, ranges.alpha_count,
                                                 sizeof(VkDrawIndexedIndirectCommand));
                    }

                    vkCmdEndRendering(cmd);
                    end_stats(cmd, *stats, GraphicsIndex::PreDepth);
                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, GraphicsStamp::PreDepthEnd);
                },
                SubmitSynchronisation{
                        .timeline_waits = cube_rotate_waits,
                });
        fs.timeline_values[stage_index(Stage::Predepth)] = predepth_val;

        const std::array culling_waits{TimelineWait{.value = fs.timeline_values[stage_index(Stage::Predepth)],
                                                    .semaphore = tl_graphics.timeline,
                                                    .stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT}};
        auto light_val = submit_stage(
                tl_compute, device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(tracy_compute.ctx, cmd, "LightCulling");

                    auto &&[cqs, css] = ctx.query_pools.get_multiple(compute_query_pool[bounded_frame_index],
                                                                     compute_stats_pool[bounded_frame_index]);

                    write_ts(cmd, *cqs, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, ComputeStamp::CullBegin);
                    begin_stats(cmd, *css, ComputeIndex::Cull);

                    const PointLightCullingPushConstants pc{
                            .ubo = aligned_frame_buffer_handle.slot_device_address(bounded_frame_index),
                            .lights = light_addr,
                            .flags = flags_addr,
                            .prefix = prefix_addr,
                            .compact = compact_addr,
                            .culled_light_count = culled_light_count_addr,
                            .light_count = light_count,
                    };

                    auto bind_and_dispatch = [&](auto &pl, u32 groups_x) {
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.layout, 0, 1, &bindless.set, 0,
                                                nullptr);

                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.pipeline);

                        vkCmdPushConstants(cmd, pl.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                           sizeof(PointLightCullingPushConstants), &pc);

                        vkCmdDispatch(cmd, groups_x, 1u, 1u);
                    };

                    VkMemoryBarrier2 mem_barrier{};
                    mem_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                    mem_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    mem_barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
                    mem_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    mem_barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;

                    VkDependencyInfo dep_info{};
                    dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dep_info.memoryBarrierCount = 1;
                    dep_info.pMemoryBarriers = &mem_barrier;

                    fill_zeros(cmd, ctx.buffers, flags_handle, prefix_handle, compact_lights_handle,
                               culled_light_count_handle);

                    vkCmdPipelineBarrier2(cmd, &dep_info);

                    const u32 gc = (light_count + THREADS_PER_GROUP - 1) / THREADS_PER_GROUP;

                    auto &&[flags, compact] = ctx.pipeline_pool.get_multiple(flags_pipeline, compact_pipeline);

                    bind_and_dispatch(*flags, gc);
                    vkCmdPipelineBarrier2(cmd, &dep_info);

                    bind_and_dispatch(*compact, gc);

                    end_stats(cmd, *css, ComputeIndex::Cull);
                    write_ts(cmd, *cqs, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, ComputeStamp::CullEnd);

                    TRACY_GPU_COLLECT(tracy_compute.ctx, cmd);
                },
                SubmitSynchronisation{.timeline_waits = culling_waits});

        fs.timeline_values[stage_index(Stage::LightCulling)] = light_val;

        const std::array gbuffer_waits{
                TimelineWait{
                        .value = fs.timeline_values[stage_index(Stage::CubeRotation)],
                        .semaphore = tl_compute.timeline,
                        .stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                },
                TimelineWait{
                        .value = fs.timeline_values[stage_index(Stage::Predepth)],
                        .semaphore = tl_graphics.timeline,
                        .stage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, // depth test
                },
        };

        auto gbuffer_val = submit_stage(
                tl_graphics, device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(tracy_graphics.ctx, cmd, "GBuffer MRT");

                    auto *ts = ctx.query_pools.get(graphics_query_pool[bounded_frame_index]);
                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, GraphicsStamp::GbufferBegin);

                    auto *mrt_pipeline = ctx.pipeline_pool.get(gbuffer_pipeline_mrt);

                    auto *g0 = ctx.textures.get(gbuffer0_handle);
                    auto *g1 = ctx.textures.get(gbuffer1_handle);
                    auto *g2 = ctx.textures.get(gbuffer2_handle);
                    auto *depth = ctx.textures.get(depth_handle);

                    g0->transition_if_not_initialised(
                            cmd, VK_IMAGE_LAYOUT_GENERAL,
                            {VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT});
                    g1->transition_if_not_initialised(
                            cmd, VK_IMAGE_LAYOUT_GENERAL,
                            {VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT});
                    g2->transition_if_not_initialised(
                            cmd, VK_IMAGE_LAYOUT_GENERAL,
                            {VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT});
                    depth->transition_if_not_initialised(cmd, VK_IMAGE_LAYOUT_GENERAL,
                                                         {VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                                                          VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT});

                    VkRenderingAttachmentInfo colors[3]{};
                    auto init_color = [&](VkRenderingAttachmentInfo &a, VkImageView view) {
                        a.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                        a.imageView = view;
                        a.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                        a.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                        a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                        a.clearValue = {.color = {.float32 = {0, 0, 0, 0}}};
                    };
                    init_color(colors[0], g0->attachment_view);
                    init_color(colors[1], g1->attachment_view);
                    init_color(colors[2], g2->attachment_view);

                    VkRenderingAttachmentInfo depth_att{};
                    depth_att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    depth_att.imageView = depth->attachment_view;
                    depth_att.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    depth_att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // keep predepth
                    depth_att.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; // no need to store
                    depth_att.clearValue = {.depthStencil = {0.0f, 0}};

                    VkRenderingInfo ri{};
                    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                    ri.renderArea = {.offset = {0, 0}, .extent = frame_extent};
                    ri.layerCount = 1;
                    ri.colorAttachmentCount = 3;
                    ri.pColorAttachments = colors;
                    ri.pDepthAttachment = &depth_att;

                    vkCmdBeginRendering(cmd, &ri);

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mrt_pipeline->pipeline);

                    auto &&[vp, sc] = viewport_scissors(frame_extent);
                    vkCmdSetViewport(cmd, 0, 1, &vp);
                    vkCmdSetScissor(cmd, 0, 1, &sc);
                    vkCmdSetDepthCompareOp(cmd,
                                           VK_COMPARE_OP_EQUAL); // matches your predepth = GEQUAL reverseZ + load depth
                    vkCmdSetCullMode(cmd, VK_CULL_MODE_BACK_BIT);
                    vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);
                    vkCmdSetDepthBounds(cmd, 0.0f, 1.0f);

                    auto &&[indirect, verts, idx, materials] =
                            ctx.buffers.get_multiple(indirect_ring.handle(), cube_mesh.vertex_buffer,
                                                     cube_mesh.index_buffer, cube_mesh.material_buffer);

                    RenderingPushConstants pc{
                            .ubo = aligned_frame_buffer_handle.slot_device_address(bounded_frame_index),
                            .transforms = cubes_transform_handle->slot_device_address(bounded_frame_index),
                            .draw_material_ids = draw_material_id_ring.slot_device_address(bounded_frame_index),
                            .materials = materials->device_address(),
                            .base_draw_id = ranges.opaque_base,
                            .sampler_index = linear_repeat.index(),
                    };

                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mrt_pipeline->layout, 0, 1,
                                            &bindless.set, 0, nullptr);

                    vkCmdPushConstants(cmd, mrt_pipeline->layout,
                                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

                    vkCmdBindIndexBuffer(cmd, idx->buffer(), 0, VK_INDEX_TYPE_UINT32);
                    VkBuffer vb = verts->buffer();
                    VkDeviceSize off = 0;
                    vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &off);

                    if (ranges.opaque_count > 0) {
                        pc.base_draw_id = ranges.opaque_base;
                        vkCmdPushConstants(cmd, mrt_pipeline->layout,
                                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc),
                                           &pc);

                        VkDeviceSize indirect_offset_bytes =
                                static_cast<VkDeviceSize>(indirect_ring.slot_offset_bytes(bounded_frame_index)) +
                                static_cast<VkDeviceSize>(ranges.opaque_base) * sizeof(VkDrawIndexedIndirectCommand);

                        vkCmdDrawIndexedIndirect(cmd, indirect->buffer(), indirect_offset_bytes, ranges.opaque_count,
                                                 sizeof(VkDrawIndexedIndirectCommand));
                    }

                    // --- Draw 2: ALPHA TESTED ---
                    if (ranges.alpha_count > 0) {
                        pc.base_draw_id = ranges.alpha_base;
                        vkCmdPushConstants(cmd, mrt_pipeline->layout,
                                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc),
                                           &pc);

                        VkDeviceSize indirect_offset_bytes =
                                static_cast<VkDeviceSize>(indirect_ring.slot_offset_bytes(bounded_frame_index)) +
                                static_cast<VkDeviceSize>(ranges.alpha_base) * sizeof(VkDrawIndexedIndirectCommand);

                        vkCmdDrawIndexedIndirect(cmd, indirect->buffer(), indirect_offset_bytes, ranges.alpha_count,
                                                 sizeof(VkDrawIndexedIndirectCommand));
                    }

                    vkCmdEndRendering(cmd);

                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, GraphicsStamp::GbufferEnd);
                },
                SubmitSynchronisation{.timeline_waits = gbuffer_waits});
        fs.timeline_values[stage_index(Stage::GBuffer)] = gbuffer_val;

        const std::array deferred_waits{
                TimelineWait{
                        .value = fs.timeline_values[stage_index(Stage::GBuffer)],
                        .semaphore = tl_graphics.timeline,
                        .stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                },
                TimelineWait{
                        .value = fs.timeline_values[stage_index(Stage::LightCulling)],
                        .semaphore = tl_compute.timeline,
                        .stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                },
        };

        auto deferred_val = submit_stage(
                tl_graphics, device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(tracy_graphics.ctx, cmd, "DeferredLighting(FS)");

                    auto &&ts = ctx.query_pools.get(graphics_query_pool[bounded_frame_index]);
                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, GraphicsStamp::DeferredBegin);

                    auto mrt_lighting = ctx.pipeline_pool.get(gbuffer_pipeline_lighting);

                    auto *g0 = ctx.textures.get(gbuffer0_handle);
                    auto *g1 = ctx.textures.get(gbuffer1_handle);
                    auto *g2 = ctx.textures.get(gbuffer2_handle);
                    auto *depth = ctx.textures.get(depth_handle);
                    auto *lit = ctx.textures.get(lit_hdr_handle);

                    g0->transition_if_not_initialised(
                            cmd, VK_IMAGE_LAYOUT_GENERAL,
                            {VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT});
                    g1->transition_if_not_initialised(
                            cmd, VK_IMAGE_LAYOUT_GENERAL,
                            {VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT});
                    g2->transition_if_not_initialised(
                            cmd, VK_IMAGE_LAYOUT_GENERAL,
                            {VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT});
                    depth->transition_if_not_initialised(cmd, VK_IMAGE_LAYOUT_GENERAL,
                                                         {VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                                          VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                                                  VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT});
                    lit->transition_if_not_initialised(
                            cmd, VK_IMAGE_LAYOUT_GENERAL,
                            {VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT});

                    const std::array<VkImageMemoryBarrier2, 5> barriers{
                            VkImageMemoryBarrier2{
                                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                    .pNext = nullptr,
                                    .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                    .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                    .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .image = g0->image,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                            },
                            VkImageMemoryBarrier2{
                                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                    .pNext = nullptr,
                                    .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                    .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                    .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .image = g1->image,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                            },
                            VkImageMemoryBarrier2{
                                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                    .pNext = nullptr,
                                    .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                    .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                    .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .image = g2->image,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                            },
                            VkImageMemoryBarrier2{
                                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                    // depth was written in predepth/gbuffer depth test
                                    .pNext = nullptr,
                                    .srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                                    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                    .srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                    .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .image = depth->image,
                                    .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
                            },
                            VkImageMemoryBarrier2{
                                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                    .pNext = nullptr,
                                    .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                                    .srcAccessMask = VK_ACCESS_2_NONE,
                                    .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                    .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .image = lit->image,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                            },
                    };

                    VkDependencyInfo dep{};
                    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dep.imageMemoryBarrierCount = static_cast<u32>(barriers.size());
                    dep.pImageMemoryBarriers = barriers.data();
                    vkCmdPipelineBarrier2(cmd, &dep);

                    VkRenderingAttachmentInfo lit_att{};
                    lit_att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    lit_att.imageView = lit->attachment_view;
                    lit_att.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    lit_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                    lit_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                    lit_att.clearValue = {.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}};

                    VkRenderingInfo ri{};
                    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                    ri.renderArea = {.offset = {0, 0}, .extent = frame_extent};
                    ri.layerCount = 1;
                    ri.colorAttachmentCount = 1;
                    ri.pColorAttachments = &lit_att;

                    vkCmdBeginRendering(cmd, &ri);

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mrt_lighting->pipeline);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mrt_lighting->layout, 0, 1,
                                            &bindless.set, 0, nullptr);

                    auto &&[vp, sc] = viewport_scissors(frame_extent);
                    vkCmdSetViewport(cmd, 0, 1, &vp);
                    vkCmdSetScissor(cmd, 0, 1, &sc);

                    DeferredLightingPushConstants pc{
                            .frame_ubo = aligned_frame_buffer_handle.slot_device_address(bounded_frame_index),
                            .point_lights = compact_addr,
                            .point_light_count = culled_light_count_addr,
                            .gbuffer0_index = gbuffer0_handle.index(),
                            .gbuffer1_index = gbuffer1_handle.index(),
                            .gbuffer2_index = gbuffer2_handle.index(),
                            .depth_index = depth_handle.index(),
                            .lit_hdr_uav_index = 0,
                            .sampler_index = linear_clamp_sampler_handle.index(),
                    };

                    vkCmdPushConstants(cmd, mrt_lighting->layout,
                                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

                    vkCmdDraw(cmd, 3, 1, 0, 0);

                    vkCmdEndRendering(cmd);

                    VkImageMemoryBarrier2 lit_to_read{
                            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                            .pNext = nullptr,
                            .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                            .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                            .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .image = lit->image,
                            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                    };

                    VkDependencyInfo dep2{};
                    dep2.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dep2.imageMemoryBarrierCount = 1;
                    dep2.pImageMemoryBarriers = &lit_to_read;
                    vkCmdPipelineBarrier2(cmd, &dep2);

                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, GraphicsStamp::DeferredEnd);
                },
                SubmitSynchronisation{.timeline_waits = deferred_waits});

        fs.timeline_values[stage_index(Stage::DeferredLighting)] = deferred_val;

        const std::array tonemap_waits{
                TimelineWait{
                        .value = fs.timeline_values[stage_index(Stage::DeferredLighting)],
                        .semaphore = tl_graphics.timeline,
                        .stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                },
        };

        auto tonemap_val = submit_stage(
                tl_graphics, device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(tracy_graphics.ctx, cmd, "Tonemapping");

                    auto &&ts = ctx.query_pools.get(graphics_query_pool[bounded_frame_index]);
                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, GraphicsStamp::TonemapBegin);

                    auto *tonemap = ctx.pipeline_pool.get(tonemap_pipeline);

                    auto &&hdr = ctx.textures.get(lit_hdr_handle);
                    auto &&ldr = ctx.textures.get(tonemapped_target_handle);

                    hdr->transition_if_not_initialised(
                            cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            {VK_ACCESS_2_SHADER_READ_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT});

                    ldr->transition(cmd, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

                    VkRenderingAttachmentInfo color_attachment{};
                    color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    color_attachment.imageView = ldr->sampled_view;
                    color_attachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                    color_attachment.clearValue = {.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}};

                    VkRenderingInfo ri{};
                    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                    ri.renderArea = {.offset = {0, 0}, .extent = {frame_extent.width, frame_extent.height}};
                    ri.layerCount = 1;
                    ri.colorAttachmentCount = 1;
                    ri.pColorAttachments = &color_attachment;

                    vkCmdBeginRendering(cmd, &ri);

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tonemap->pipeline);

                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tonemap->layout, 0, 1, &bindless.set,
                                            0, nullptr);

                    float exposure = 1.0f;
                    TonemapPushConstants pc{
                            .exposure = exposure,
                            .image_index = lit_hdr_handle.index(),
                            .sampler_index = linear_clamp_sampler_handle.index(),
                    };


                    auto &&[vp, sc] = viewport_scissors(frame_extent);
                    vkCmdSetViewport(cmd, 0, 1, &vp);
                    vkCmdSetScissor(cmd, 0, 1, &sc);

                    vkCmdPushConstants(cmd, tonemap->layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                       0, sizeof(pc), &pc);

                    vkCmdDraw(cmd, 3, 1, 0, 0);

                    vkCmdEndRendering(cmd);

                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, GraphicsStamp::TonemapEnd);
                },
                SubmitSynchronisation{.timeline_waits = tonemap_waits});

        fs.timeline_values[stage_index(Stage::Tonemapping)] = tonemap_val;

        const std::array present_timeline_waits{
                TimelineWait{
                        .value = fs.timeline_values[stage_index(Stage::Tonemapping)],
                        .semaphore = tl_graphics.timeline,
                },
        };

        const std::array present_binary_waits{BinaryWait{
                .semaphore = frame_sync.image_available,
                .stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        }};

        const std::array present_binary_signals{frame_sync.render_finished};

        auto swapchain_val = submit_stage(
                tl_graphics, device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(tracy_graphics.ctx, cmd, "BlitToSwapchain");

                    auto &&tonemapped = ctx.textures.get(tonemapped_target_handle);
                    const auto dst_image = swapchain.image(swap_image_index);
                    const auto src_image = tonemapped->image;

                    auto &&[ts, stats] = ctx.query_pools.get_multiple(graphics_query_pool[bounded_frame_index],
                                                                      graphics_stats_pool[bounded_frame_index]);
                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, GraphicsStamp::PresentBegin);
                    begin_stats(cmd, *stats, GraphicsIndex::Present);

                    // 1. Transition Barriers
                    const std::array barriers{VkImageMemoryBarrier2{
                                                      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                                      .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                                      .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                                      .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                      .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                                                      .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                                      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                      .image = src_image,
                                                      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                                              },
                                              VkImageMemoryBarrier2{
                                                      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                                      .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                                                      .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                      .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                                      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                      .image = dst_image,
                                                      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                                              }};

                    VkDependencyInfo dep_info{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                              .imageMemoryBarrierCount = static_cast<u32>(barriers.size()),
                                              .pImageMemoryBarriers = barriers.data()};
                    vkCmdPipelineBarrier2(cmd, &dep_info);

                    // 2. The Blit Operation
                    // Blit automatically handles the R8G8B8A8 -> B8G8R8A8 conversion!
                    VkImageBlit region{};
                    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    region.srcOffsets[0] = {0, 0, 0};
                    region.srcOffsets[1] = {static_cast<int32_t>(frame_extent.width),
                                            static_cast<int32_t>(frame_extent.height), 1};

                    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    region.dstOffsets[0] = {0, 0, 0};
                    region.dstOffsets[1] = {static_cast<int32_t>(frame_extent.width),
                                            static_cast<int32_t>(frame_extent.height), 1};

                    vkCmdBlitImage(cmd, src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst_image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region,
                                   VK_FILTER_LINEAR); // Linear filtering in case of resize

                    // 3. Final Transition to Present
                    VkImageMemoryBarrier2 present_barrier{
                            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                            .dstStageMask = VK_PIPELINE_STAGE_2_NONE,
                            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                            .image = dst_image,
                            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                    };

                    VkDependencyInfo end_dep{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                             .imageMemoryBarrierCount = 1,
                                             .pImageMemoryBarriers = &present_barrier};
                    vkCmdPipelineBarrier2(cmd, &end_dep);

                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, GraphicsStamp::PresentEnd);
                    end_stats(cmd, *stats, GraphicsIndex::Present);
                },
                SubmitSynchronisation{
                        .timeline_waits = present_timeline_waits,
                        .binary_waits = present_binary_waits,
                        .binary_signals = present_binary_signals,
                });

        fs.frame_done_value = swapchain_val;

        //  throttle(tl_graphics, ctx.get_device());
        //  throttle(tl_compute, ctx.get_device());

        const auto completed = std::min(tl_compute.completed, tl_graphics.completed);
        ctx.destroy_queue.retire(completed);
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(frame_end - start_time).count();
        stats.add_sample(ms);

        const VkResult present_res = swapchain.present(graphics_queue, swap_image_index, frame_sync.render_finished);
        FrameMark;
        if (present_res == VK_ERROR_OUT_OF_DATE_KHR || present_res == VK_SUBOPTIMAL_KHR) {
            auto result = swapchain.recreate(current_extent(window));
            if (!result)
                vk_check(result.error());
        } else {
            vk_check(present_res);
        }

        frame_index++;
    }

    info("Light count {}", opts.light_count);
    info("frames: {}", stats.samples.size());
    info("mean/frametime:   {:.3f} ms", stats.mean);
    info("median: {:.3f} ms", stats.median());
    info("stddev: {:.3f} ms", stats.stddev_sample());
    info("quartiles: {}", stats.quartiles());
    info("Total: {:.3f} s", stats.total() / 1000.0F);

    info("GPU compute mean:   {:.3f} ms", gpu_compute_ms.avg());
    info("GPU compute p95:    {:.3f} ms", gpu_compute_ms.p95());
    info("GPU graphics mean:  {:.3f} ms", gpu_graphics_ms.avg());
    info("GPU graphics p95:   {:.3f} ms", gpu_graphics_ms.p95());

#ifdef HAS_IMAGE_WRITERS
    if (!opts.disable_output_images) {
        const auto &&[oth, gbuffer0, gbuffer1, gbuffer2, ph] = ctx.textures.get_multiple(
                lit_hdr_handle, gbuffer0_handle, gbuffer1_handle, gbuffer2_handle, perlin_handle);
        std::filesystem::path output_dir = "output";
        std::filesystem::create_directory(output_dir);
        const auto as_string = output_dir.string();
        ZoneScopedNC("batch_write_images", 0xFF00AA);
        std::vector requests{image_operations::ImageWriteRequest{oth, std::format("{}/output.png", as_string)},
                             image_operations::ImageWriteRequest{
                                     gbuffer0,
                                     std::format("{}/gbuffer0.png", as_string),
                             },
                             image_operations::ImageWriteRequest{
                                     gbuffer1,
                                     std::format("{}/gbuffer1.png", as_string),
                             },
                             image_operations::ImageWriteRequest{
                                     gbuffer2,
                                     std::format("{}/gbuffer2.png", as_string),
                             },
                             image_operations::ImageWriteRequest{
                                     ph,
                                     std::format("{}/perlin.png", as_string),
                             }};
        image_operations::write_batch_to_disk(
                allocator, requests, [](float progress) { info("Image write progress: {:.2f} %", progress * 100.0f); });
    }
#endif
    vkDeviceWaitIdle(device);

    pipeline_cache.reset();
    ctx.clear_all();

    compiler.reset();

    watcher.reset();
    listeners.clear();

    ctx.destroy_queue.retire(UINT64_MAX);

    tracy_compute.shutdown();
    tracy_graphics.shutdown();

    destruction::global_command_context(command_context);
    destruction::bindless_set(device, bindless);
    destruction::timelines(device, tl_graphics, tl_transfer, tl_compute);
    destruction::allocator(allocator);
    destruction::swapchain(swapchain);
    destruction::wsi(instance.instance, surface, window);
    destruction::device(device);

    return 0;
}


auto main(int argc, char **argv) -> int {
    if (auto init = glfwInit(); init != GLFW_TRUE) {
        error("Could not initialize GLFW");
        return 1;
    }

    auto opts = parse_cli(argc, argv);

    uint32_t count{};
    const char **extensions_raw = glfwGetRequiredInstanceExtensions(&count);
    std::vector<std::string_view> extensions(extensions_raw, extensions_raw + count);

    // Priority: opts.validation_layers overrides IS_RELEASE
    // - If opts.validation_layers is explicitly set, use that
    // - Otherwise, enable validation in debug builds, disable in release builds
    bool enable_validation = opts.validation_layers.value_or(!static_cast<bool>(IS_RELEASE));

    InstanceWithDebug instance;
    if (enable_validation) {
        // With validation layers
        instance = create_instance_with_debug(debug_callback, extensions);
    } else {
        // No validation
        auto raw_instance = create_instance(extensions);
        instance.instance = raw_instance;
        instance.messenger = VK_NULL_HANDLE;
    }

    auto execute_result = execute(opts, instance);
    if (!execute_result) {
        error("Failed to execute: {}", execute_result.error());
        return 1;
    }

    destruction::instance(instance);
    volkFinalize();
    glfwTerminate();
    info("Bindless headless setup and teardown completed successfully.");
    return 0;
}
