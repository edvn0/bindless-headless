#include "AlignedRingBuffer.hxx"
#include "ArgumentParse.hxx"
#include "BindlessHeadless.hxx"
#include "BindlessSet.hxx"
#include "Buffer.hxx"
#include "Compiler.hxx"
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
#include <chrono>
#include <efsw/efsw.hpp>
#include <execution>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/packing.hpp>
#include <iostream>
#include <ranges>
#include <thread>
#include <tracy/Tracy.hpp>

#include "3PP/PerlinNoise.hpp"
#include "Profiler.hxx"
#include "vulkan/vulkan_core.h"

#include <Windows.h>

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
            return VkSampleCountFlagBits{}; // treat 0 as "auto" if you want
        default:
            return VK_SAMPLE_COUNT_1_BIT; // or error out
    }
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

struct Mesh {
    BufferHandle vertex_buffer;
    BufferHandle index_buffer;
    std::string name;

    template<typename Vert, typename Idx>
        requires std::is_trivial_v<Vert> && (std::is_same_v<Idx, u32> || std::is_same_v<Idx, u16>)
    static auto create(RenderContext &ctx, std::span<Vert> vertices, std::span<Idx> indices, std::string_view name)
            -> Mesh {
        const auto vertex_name = std::format("{}_vertices", name);
        const auto index_name = std::format("{}_indices", name);
        auto vertex_buffer =
                ctx.buffers.create(Buffer::from_slice<Vert>(ctx.allocator,
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, vertices, vertex_name)
                                           .value());
        auto index_buffer =
                ctx.buffers.create(Buffer::from_slice<Idx>(ctx.allocator,VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, indices, index_name)
                                           .value());
        return Mesh{
                .vertex_buffer = vertex_buffer,
                .index_buffer = index_buffer,
                .name = std::string{name},
        };
    }
};

struct Vertex {
    glm::vec3 position;
    uint32_t normal; // packed 10_10_10_2
    uint32_t uvs; // packed 8_8_8_8
};
static_assert(std::is_trivial_v<Vertex>);
static_assert(sizeof(Vertex) == 20);


// Generates a cube with 24 vertices and 36 indices using GLM packing
inline void generate_cube(std::array<Vertex, 24> &out_vertices, std::array<uint16_t, 36> &out_indices) {
    static_assert(std::is_trivial_v<Vertex>);

    struct Face {
        glm::vec3 normal;
        glm::vec3 v[4];
    };

    uint32_t vert_index = 0;
    uint32_t idx_index = 0;

    for (uint32_t f = 0; f < 6; ++f) {
        constexpr std::array<Face, 6> faces = {
                // -Z
                Face{{0, 0, -1}, {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1}}},
                // +Z
                {{0, 0, 1}, {{-1, -1, 1}, {-1, 1, 1}, {1, 1, 1}, {1, -1, 1}}},
                // -X
                {{-1, 0, 0}, {{-1, -1, 1}, {-1, -1, -1}, {-1, 1, -1}, {-1, 1, 1}}},
                // +X
                {{1, 0, 0}, {{1, -1, -1}, {1, -1, 1}, {1, 1, 1}, {1, 1, -1}}},
                // -Y
                {{0, -1, 0}, {{-1, -1, 1}, {1, -1, 1}, {1, -1, -1}, {-1, -1, -1}}},
                // +Y
                {{0, 1, 0}, {{-1, 1, -1}, {1, 1, -1}, {1, 1, 1}, {-1, 1, 1}}},
        };
        u32 base = vert_index;

        // Pack normal using GLM
        glm::vec4 normal4(faces.at(f).normal, 0.0f); // w = 0 for the extra 2 bits
        u32 packed_normal = glm::packUnorm3x10_1x2(normal4);

        // Pack full UV rect
        u32 packed_uv = (0x00u) | (0x00u << 8) | (0xFFu << 16) | (0xFFu << 24); // full 0..1

        for (int v = 0; v < 4; ++v) {
            out_vertices[vert_index++] = {faces[f].v[v], packed_normal, packed_uv};
        }

        // Two triangles per face
        auto as_u16 = static_cast<u16>(base);
        out_indices[idx_index++] = as_u16 + 0;
        out_indices[idx_index++] = as_u16 + 1;
        out_indices[idx_index++] = as_u16 + 2;

        out_indices[idx_index++] = as_u16 + 2;
        out_indices[idx_index++] = as_u16 + 3;
        out_indices[idx_index++] = as_u16 + 0;
    }
}

enum class PipelineStats : u32 {
    InputAssemblyVertices = 0,
    InputAssemblyPrimitives = 1,
    VertexShaderInvocations = 2,
    ClippingInvocations = 3,
    ClippingPrimitives = 4,
    FragmentShaderInvocations = 5,
    ComputeShaderInvocations = 6,
    Count = 7,
};

constexpr u32 pipeline_stats_query_count = 1;

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
                                         sizeof(stats), stats.data(), sizeof(u64),
                                         VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

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
                                         sizeof(stats), stats.data(), sizeof(u64),
                                         VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

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

    std::array<FrustumPlane, 6> planes;
    planes[0].plane = compute_plane(v[0], v[2], v[4]); // Left
    planes[1].plane = compute_plane(v[1], v[5], v[3]); // Right
    planes[2].plane = compute_plane(v[0], v[4], v[1]); // Bottom
    planes[3].plane = compute_plane(v[2], v[3], v[6]); // Top
    planes[4].plane = compute_plane(v[0], v[1], v[2]); // Near
    planes[5].plane = compute_plane(v[4], v[6], v[5]); // Far

    return planes;
};

struct FrameUBO {
    glm::mat4 view;
    glm::mat4 projection;
    glm::mat4 view_projection;
    glm::mat4 inv_projection;
    glm::vec4 camera_position;
    std::array<FrustumPlane, 6> frustum_planes; // left, right, bottom, top, near, far
    glm::vec4 sun_direction_intensity;
};


auto generate_perlin(auto w, auto h) -> std::vector<std::uint8_t> {
    std::vector<std::uint8_t> data;
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
                               VkDebugUtilsMessageTypeFlagsEXT type,
                               const VkDebugUtilsMessengerCallbackDataEXT *callback_data, void *) {
    if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        error("Validation layer: {}", callback_data->pMessage);
    }

    if (type & VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT) {
        info("What is this?: {}", callback_data->pMessage);
    }
    return VK_FALSE;
}


class UpdateListener : public efsw::FileWatchListener {
public:
    void handleFileAction(efsw::WatchID, const std::string &dir, const std::string &filename, efsw::Action action,
                          std::string oldFilename) override {
        switch (action) {
            case efsw::Actions::Add:
                std::cout << "DIR (" << dir << ") FILE (" << filename << ") has event Added" << std::endl;
                break;
            case efsw::Actions::Delete:
                std::cout << "DIR (" << dir << ") FILE (" << filename << ") has event Delete" << std::endl;
                break;
            case efsw::Actions::Modified:
                std::cout << "DIR (" << dir << ") FILE (" << filename << ") has event Modified" << std::endl;
                break;
            case efsw::Actions::Moved:
                std::cout << "DIR (" << dir << ") FILE (" << filename << ") has event Moved from (" << oldFilename
                          << ")" << std::endl;
                break;
            default:
                std::cout << "Should never happen!" << std::endl;
        }
    }
};


static MaybeNoOp<PFN_vkCmdDrawMeshTasksIndirectEXT> draw_mesh{};

struct Deleter {
    template<typename T>
    auto operator()(T *t) noexcept -> void {
        delete t;
    }
};

auto execute(int argc, char **argv) -> int {
    std::unique_ptr<efsw::FileWatcher, Deleter> watcher(new efsw::FileWatcher(false), Deleter{});
    std::unordered_map<std::string, std::unique_ptr<efsw::FileWatchListener, Deleter>> listeners;
    listeners["update"] = std::unique_ptr<efsw::FileWatchListener, Deleter>(new UpdateListener(), Deleter{});

    std::ignore = watcher->addWatch("shaders", listeners["update"].get(), true,
                                    {efsw::WatcherOption(efsw::Option::WinBufferSize, 128 * 1024)});

    watcher->watch();

    if (auto init = glfwInit(); init != GLFW_TRUE) {
        error("Could not initialize GLFW");
        return 1;
    }

    auto opts = parse_cli(argc, argv);

    auto compiler = Compiler{};

    constexpr bool is_release = static_cast<bool>(IS_RELEASE);

    uint32_t count{};
    const char **extensions_raw = glfwGetRequiredInstanceExtensions(&count);
    std::vector<std::string_view> extensions(extensions_raw, extensions_raw + count);

    auto instance = create_instance_with_debug(debug_callback, extensions, is_release);
    auto could_choose = pick_physical_device(instance.instance);
    if (!could_choose) {
        return 1;
    }

    if (auto name = vkGetInstanceProcAddr(instance.instance, "vkCmdDrawMeshTasksIndirectEXT");
        name && draw_mesh.empty()) {
        draw_mesh = std::bit_cast<PFN_vkCmdDrawMeshTasksIndirectEXT>(name);
    }

    auto &&[physical_device, graphics_index, compute_index] = *could_choose;
    auto &&[device, graphics_queue, compute_queue] = create_device(physical_device, graphics_index, compute_index);

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


    auto cache_path = pipeline_cache_path(argc, argv);
    auto pipeline_cache = std::make_unique<PipelineCache>(device, opts.pipeline_cache_dir);

    auto command_context = create_global_cmd_context(device, graphics_queue, graphics_index);

    std::array<const std::string_view, 2> names = {"LightFlagsCS", "LightCompactCS"};
    std::array<ReflectionData, names.size()> reflection_data = {};
    auto culling_code = compiler.compile_from_file("shaders/light_cull_compact_modern.slang", std::span(names),
                                                   std::span(reflection_data));

    std::array<const std::string_view, 2> point_light_names = {"main_vs", "main_fs"};
    std::array<ReflectionData, point_light_names.size()> point_light_reflection = {};
    auto point_light_code = compiler.compile_from_file("shaders/point_light.slang", std::span(point_light_names),
                                                       std::span(point_light_reflection));

    std::array<const std::string_view, 2> predepth_names{"main_vs", "main_fs"};
    std::array<ReflectionData, predepth_names.size()> predepth_reflection{};
    auto predepth_code = compiler.compile_from_file("shaders/predepth.slang", std::span(predepth_names),
                                                    std::span(predepth_reflection));

    std::array<const std::string_view, 2> tonemap_names{"vs_main", "fs_main"};
    std::array<ReflectionData, tonemap_names.size()> tonemap_reflection{};
    auto tonemap_code = compiler.compile_from_file("shaders/tonemap.slang", std::span(tonemap_names),
                                                   std::span(tonemap_reflection));

    auto allocator = create_allocator(instance.instance, physical_device, device);

    auto tl_compute = create_compute_timeline(device, compute_queue, compute_index);
    auto tl_graphics = create_graphics_timeline(device, graphics_queue, graphics_index);

    BindlessSet bindless{};
    bindless.init(device, query_bindless_caps(physical_device), 8u, 8u, 8u, 0u);
    bindless.grow_if_needed(300u, 40u, 32u, 8u);


    const VkSampleCountFlagBits requested = msaa_from_cli(opts.msaa);
    const VkSampleCountFlagBits msaa_samples = clamp_msaa_samples(physical_device, requested);
    info("MSAA requested: {}, Engine supplied: {}", static_cast<u32>(requested), static_cast<u32>(msaa_samples));


    auto &&[flags_pipeline, compact_pipeline] = create_compute_pipelines(device, *pipeline_cache, bindless.layout,
                                                                         std::span(culling_code), std::span(names));

    auto point_light_pipeline =
            create_mesh_pipeline(device, *pipeline_cache, bindless.layout, point_light_code.at(0),
                                 point_light_code.at(1), VK_FORMAT_R32G32B32A32_SFLOAT, msaa_samples);
    auto predepth_pipeline = create_predepth_pipeline(device, *pipeline_cache, bindless.layout, predepth_code.at(0),
                                                      predepth_code.at(1), VK_FORMAT_D32_SFLOAT, msaa_samples);

    auto tonemap_pipeline = create_tonemap_pipeline(device, *pipeline_cache, bindless.layout, tonemap_code.at(0),
                                                    tonemap_code.at(1), "vs_main", "fs_main", VK_FORMAT_R8G8B8A8_SRGB);

    RenderContext ctx{
            .allocator = allocator,
            .bindless_set = &bindless,
    };


    std::array<QueryPoolHandle, frames_in_flight> compute_query_pool{};
    std::array<QueryPoolHandle, frames_in_flight> graphics_query_pool{};
    std::array<QueryPoolHandle, frames_in_flight> graphics_stats_pool{};
    std::array<QueryPoolHandle, frames_in_flight> compute_stats_pool{};
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical_device, &props);
        const auto timestamp_period_ns = static_cast<double>(props.limits.timestampPeriod);

        for (u32 fi = 0; fi < frames_in_flight; ++fi) {
            VkQueryPoolCreateInfo qpci{.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                       .pNext = nullptr,
                                       .flags = 0,
                                       .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                       .queryCount = query_count,
                                       .pipelineStatistics = 0};

            VkQueryPool qpc = VK_NULL_HANDLE;
            vk_check(vkCreateQueryPool(device, &qpci, nullptr, &qpc));
            compute_query_pool[fi] = ctx.create_query_pool(QueryPoolState{
                    .pool = qpc, .query_count = query_count, .timestamp_period_ns = timestamp_period_ns});
            set_debug_name(device, VK_OBJECT_TYPE_QUERY_POOL, qpc,
                           std::format("compute_timestamp_query_pool_frame_{}", fi));

            VkQueryPool qpg = VK_NULL_HANDLE;
            vk_check(vkCreateQueryPool(device, &qpci, nullptr, &qpg));
            graphics_query_pool[fi] = ctx.create_query_pool(QueryPoolState{
                    .pool = qpg, .query_count = query_count, .timestamp_period_ns = timestamp_period_ns});
            set_debug_name(device, VK_OBJECT_TYPE_QUERY_POOL, qpg,
                           std::format("graphics_timestamp_query_pool_frame_{}", fi));

            VkQueryPoolCreateInfo stats_qpci{
                    .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS,
                    .queryCount = pipeline_stats_query_count,
                    .pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT |
                                          VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT |
                                          VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT |
                                          VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT |
                                          VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
                                          VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT |
                                          VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT |
                                          VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT |
                                          VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT,
            };

            VkQueryPool stats_pool = VK_NULL_HANDLE;
            vk_check(vkCreateQueryPool(device, &stats_qpci, nullptr, &stats_pool));
            graphics_stats_pool[fi] = ctx.create_query_pool(QueryPoolState{
                    .pool = stats_pool,
                    .query_count = pipeline_stats_query_count,
                    .timestamp_period_ns = 0.0, // Not used for stats
            });
            set_debug_name(device, VK_OBJECT_TYPE_QUERY_POOL, stats_pool,
                           std::format("graphics_stats_query_pool_frame_{}", fi));

            // For compute statistics
            VkQueryPoolCreateInfo compute_stats_qpci{
                    .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS,
                    .queryCount = pipeline_stats_query_count,
                    .pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT,
            };

            VkQueryPool compute_stats = VK_NULL_HANDLE;
            vk_check(vkCreateQueryPool(device, &compute_stats_qpci, nullptr, &compute_stats));
            compute_stats_pool[fi] = ctx.create_query_pool(QueryPoolState{
                    .pool = compute_stats,
                    .query_count = pipeline_stats_query_count,
                    .timestamp_period_ns = 0.0,
            });
            set_debug_name(device, VK_OBJECT_TYPE_QUERY_POOL, compute_stats,
                           std::format("compute_stats_query_pool_frame_{}", fi));
        }
    }

    ctx.create_texture(
            create_offscreen_target(allocator, opts.width, opts.height, VK_FORMAT_R8G8B8A8_UNORM, {}, "white-texture"));
    ctx.create_texture(
            create_offscreen_target(allocator, opts.width, opts.height, VK_FORMAT_R8G8B8A8_UNORM, {}, "black-texture"));

    const auto noise = generate_perlin(2048, 2048);
    auto perlin_handle = ctx.create_texture(create_image_from_span_v2(
            allocator, command_context, 2048u, 2048u, VK_FORMAT_R8_UNORM, std::span{noise}, "perlin_noise"));


    TextureHandle offscreen_target_handle;
    TextureHandle msaa_offscreen_target_handle;
    TextureHandle tonemapped_target_handle;
    TextureHandle offscreen_depth_target_handle;
    TextureHandle msaa_offscreen_depth_target_handle;

    ctx.create_sampler(
            VkSamplerCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .magFilter = VK_FILTER_LINEAR,
                    .minFilter = VK_FILTER_LINEAR,
                    .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                    .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    .mipLodBias = 0.0f,
                    .anisotropyEnable = VK_FALSE,
                    .maxAnisotropy = 1.0f,
                    .compareEnable = VK_FALSE,
                    .compareOp = VK_COMPARE_OP_ALWAYS,
                    .minLod = 0.0f,
                    .maxLod = VK_LOD_CLAMP_NONE,
                    .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
                    .unnormalizedCoordinates = VK_FALSE,
            },
            "linear_repeat");

    auto linear_clamp_sampler_handle = ctx.create_sampler(
            VkSamplerCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .magFilter = VK_FILTER_LINEAR,
                    .minFilter = VK_FILTER_LINEAR,
                    .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                    .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, // Changed
                    .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, // Changed
                    .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, // Changed
                    .mipLodBias = 0.0f,
                    .anisotropyEnable = VK_FALSE,
                    .maxAnisotropy = 1.0f,
                    .compareEnable = VK_FALSE,
                    .compareOp = VK_COMPARE_OP_ALWAYS,
                    .minLod = 0.0f,
                    .maxLod = VK_LOD_CLAMP_NONE,
                    .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
                    .unnormalizedCoordinates = VK_FALSE,
            },
            "linear_clamp");

    ctx.create_sampler(create_sampler(allocator,
                                      VkSamplerCreateInfo{
                                              .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                              .magFilter = VK_FILTER_LINEAR,
                                              .minFilter = VK_FILTER_LINEAR,
                                              .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                              .mipLodBias = 0.0f,
                                              .anisotropyEnable = false,
                                              .compareEnable = false,
                                              .minLod = 0.0f,
                                              .maxLod = VK_LOD_CLAMP_NONE,
                                              .unnormalizedCoordinates = false,
                                      },
                                      "noise_sampler"));

    bindless.repopulate_if_needed(ctx.textures, ctx.samplers);

    std::array<FrameState, frames_in_flight> frames{};

    struct PointLight {
        std::array<float, 4> position_radius;
        std::array<float, 4> colour_intensity;
    };

    auto all_point_lights = std::vector<PointLight>(opts.light_count);
    auto all_point_lights_zero = std::vector<PointLight>(opts.light_count);
    auto light_count = static_cast<u32>(all_point_lights.size());
    constexpr u32 threads_per_group = 64u;
    auto group_count = (light_count + threads_per_group - 1u) / threads_per_group;

    constexpr auto world_size = 200.F;

    auto rng = std::default_random_engine{
            static_cast<u32>(std::chrono::high_resolution_clock::now().time_since_epoch().count())};

    auto distrib = std::uniform_real_distribution{-world_size, world_size};
    auto intensity_distrib = std::uniform_real_distribution{0.1F, 10.0F};

    for (u32 idx = 0; idx < light_count; ++idx) {
        auto t = static_cast<float>(idx) / static_cast<float>(light_count);
        auto &[position_radius, colour_intensity] = all_point_lights[idx];

        position_radius = {distrib(rng), distrib(rng), distrib(rng), 5.0F};
        colour_intensity = {t, 1.0f - t, 0.5f, intensity_distrib(rng)};
    }

    struct Cube {
        std::array<float, 4> position_radius;
        std::array<float, 4> colour_intensity;
    };

    auto cubes = std::vector<Cube>(opts.light_count / 10);
    for (u32 idx = 0; idx < cubes.size(); ++idx) {
        auto t = static_cast<float>(idx) / static_cast<float>(cubes.size());
        auto &[position_radius, colour_intensity] = cubes[idx];

        position_radius = {distrib(rng), distrib(rng), distrib(rng), 5.0F};
        colour_intensity = {t, 1.0f - t, 0.5f, intensity_distrib(rng)};
    }

    auto point_light_handle = ctx.buffers.create(
            Buffer::from_slice<PointLight>(allocator,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, all_point_lights, "point_light")
                    .value());

    auto culled_light_count_handle =
            ctx.buffers.create(Buffer::from_value<u32>(allocator,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 0u, "culled_point_light_count")
                                       .value());

    auto mapped_to_transforms = cubes | std::views::transform([](const Cube &cube) {
                                    glm::mat4 transform = glm::translate(
                                            glm::mat4(1.0f), glm::vec3{cube.position_radius[0], cube.position_radius[1],
                                                                       cube.position_radius[2]});
                                    transform = glm::scale(transform, glm::vec3{cube.position_radius[3]});
                                    return transform;
                                }) |
                                std::ranges::to<std::vector<glm::mat4>>();
    std::vector<glm::vec3> random_factors(mapped_to_transforms.size());

    for (auto &f: random_factors) {
        std::uniform_real_distribution<float> dist(0.5f, 2.0f);
        f = glm::vec3(dist(rng), dist(rng), dist(rng));
    }
    auto cubes_transform_handle =
            AlignedRingBuffer<glm::mat4>::create(ctx, mapped_to_transforms.size(), 0u, "transforms");

    auto instance_count = static_cast<u32>(cubes.size());

    auto cube_vertices = std::array<Vertex, 24>{};
    auto cube_indices = std::array<u16, 36>{};
    generate_cube(cube_vertices, cube_indices);
    auto cube_prepdepth_vertices = cube_vertices | std::views::transform([](auto &v) { return v.position; }) |
                                   std::ranges::to<std::vector<glm::vec3>>();

    auto cube_mesh = Mesh::create<Vertex, u16>(ctx, std::span(cube_vertices), std::span(cube_indices), "cube");
    auto cube_predepth_mesh =
            Mesh::create<glm::vec3, u16>(ctx, std::span(cube_prepdepth_vertices), std::span(cube_indices), "cube");

    std::vector zeros_lights(light_count, 0u);
    std::vector zeros_groups(group_count, 0u);
    auto flags_handle =
            ctx.buffers.create(Buffer::from_slice<u32>(allocator,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT, zeros_lights, "light_flags")
                                       .value());
    auto prefix_handle =
            ctx.buffers.create(Buffer::from_slice<u32>(allocator,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT, zeros_lights, "light_prefix")
                                       .value());

    auto compact_lights_handle = ctx.buffers.create(
            Buffer::from_slice<PointLight>(allocator,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,all_point_lights_zero, "compact_lights")
                    .value());

    auto aligned_frame_buffer_handle = AlignedRingBuffer<FrameUBO>::create(ctx, "aligned_frame_ubo_buffer").value();

    auto light_addr = ctx.device_address(point_light_handle);
    auto flags_addr = ctx.device_address(flags_handle);
    auto prefix_addr = ctx.device_address(prefix_handle);
    auto compact_addr = ctx.device_address(compact_lights_handle);
    auto culled_light_count_addr = ctx.device_address(culled_light_count_handle);

    auto stats = FrameStats{};
    FrameStats gpu_compute_ms{};
    FrameStats gpu_graphics_ms{};

    auto read_timestamp_ms = [&](const auto h) -> std::optional<double> {
        const auto *qs = ctx.query_pools.get(h);
        if (!qs)
            return std::nullopt;

        std::array<u64, 2> stamps = {};
        const auto r = vkGetQueryPoolResults(device, qs->pool, 0, 2, sizeof(stamps), stamps.data(), sizeof(u64),
                                             VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (r != VK_SUCCESS)
            return std::nullopt;

        const u64 dt_ticks = stamps[1] - stamps[0];
        const double dt_ns = static_cast<double>(dt_ticks) * qs->timestamp_period_ns;
        constexpr auto ns_to_ms_factor = 1e-6;
        return dt_ns * ns_to_ms_factor;
    };

    struct WindowData {
        bool resized{false};
    } wd;
    auto current_extent = [](GLFWwindow *win) {
        int fbw{0};
        int fbh{0};
        glfwGetFramebufferSize(win, &fbw, &fbh);
        return VkExtent2D{.width = static_cast<u32>(std::max(fbw, 0)), .height = static_cast<u32>(std::max(fbh, 0))};
    };

    glfwSetWindowUserPointer(window, &wd);
    glfwSetKeyCallback(window, [](auto w, auto k, auto, auto, auto) {
        if (k == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(w, GLFW_TRUE);
        }
    });
    glfwSetWindowSizeCallback(window, [](auto w, auto, auto) {
        auto &data = *static_cast<WindowData *>(glfwGetWindowUserPointer(w));
        data.resized = true;
    });
    glfwSetFramebufferSizeCallback(window, [](auto w, auto, auto) {
        auto &data = *static_cast<WindowData *>(glfwGetWindowUserPointer(w));
        data.resized = true;
    });

    glfwShowWindow(window);
    glfwFocusWindow(window);

    double time_total = 0.0f;

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
                                                                                   const ResizeContext &resize_ctx) {
            const auto old_color = offscreen_target_handle;
            const auto old_depth = offscreen_depth_target_handle;
            const auto old_msaa_color = msaa_offscreen_target_handle;
            const auto old_msaa_depth = msaa_offscreen_depth_target_handle;

            offscreen_target_handle = ctx.create_texture(create_offscreen_target(
                    allocator, e.width, e.height, VK_FORMAT_R32G32B32A32_SFLOAT, {}, "offscreen"));
            offscreen_depth_target_handle = ctx.create_texture(create_depth_target(
                    allocator, e.width, e.height, VK_FORMAT_D32_SFLOAT, msaa_samples, false, "offscreen_depth"));
            msaa_offscreen_target_handle = ctx.create_texture(
                    create_offscreen_target(allocator, e.width, e.height, VK_FORMAT_R32G32B32A32_SFLOAT, msaa_samples,
                                            {.sampled_storage_transfer = 0b000}, "msaa_offscreen"));
            msaa_offscreen_depth_target_handle = ctx.create_texture(create_depth_target(
                    allocator, e.width, e.height, VK_FORMAT_D32_SFLOAT, msaa_samples, false, "msaa_offscreen_depth"));


            destroy(ctx, old_color, resize_ctx.retire_value);
            destroy(ctx, old_depth, resize_ctx.retire_value);
            destroy(ctx, old_msaa_color, resize_ctx.retire_value);
            destroy(ctx, old_msaa_depth, resize_ctx.retire_value);
        });

        const auto uniforms_node =
                resize_graph.add_node("frame_ubo_camera", [&](VkExtent2D e, const ResizeContext &resize_context) {
                    const float aspect_ratio = static_cast<float>(e.width) / static_cast<float>(e.height);

                    FrameUBO ubo_data{};
                    const auto camera_pos = glm::vec3{15, 10, -20};
                    ubo_data.view = glm::lookAt(camera_pos, glm::vec3{0, 0, 0}, glm::vec3{0, 1, 0});
                    ubo_data.projection = PerspectiveRH_ReverseZ_Inf(glm::radians(70.0f), aspect_ratio, 1.F);
                    const auto frustum_projection =
                            glm::perspectiveFov(glm::radians(70.0f), static_cast<float>(e.width),
                                                static_cast<float>(e.height), 0.1F, 1000.F);
                    ubo_data.inv_projection = glm::inverse(frustum_projection);
                    ubo_data.view_projection = ubo_data.projection * ubo_data.view;
                    ubo_data.camera_position = glm::vec4{camera_pos, 1.0};
                    const auto planes = extract_frustum_planes(ubo_data.inv_projection);
                    ubo_data.frustum_planes = {planes[0], planes[1], planes[2], planes[3], planes[4], planes[5]};

                    aligned_frame_buffer_handle.write_all_slots(resize_context.ctx, ubo_data);
                });


        resize_graph.add_dependency(tonemapped_node, offscreen_node);
        resize_graph.add_dependency(offscreen_node, swapchain_node);
        resize_graph.add_dependency(uniforms_node, swapchain_node);
    }

    resize_graph.rebuild(last_extent, ResizeContext{
                                              .ctx = ctx,
                                              .retire_value = 0,
                                      });

    u64 frame_index{};
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        const u64 completed_now = std::min(tl_compute.completed, tl_graphics.completed);

        if (const auto extent = current_extent(window);
            extent.width != last_extent.width || extent.height != last_extent.height) {
            if (extent.width == 0 || extent.height == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            last_extent = extent;


            resize_graph.rebuild(extent, ResizeContext{
                                                 .ctx = ctx,
                                                 .retire_value = completed_now,
                                         });

            continue;
        }

        const auto frame_extent = swapchain.extent();
        auto start_time = std::chrono::high_resolution_clock::now();
        const auto bounded_frame_index = static_cast<u32>(frame_index % frames_in_flight);

        {
            const double t = glm::radians(73.0f);
            const glm::vec3 sun_dir = glm::normalize(glm::vec3(cos(t), sin(t), -0.4f));

            auto sun_direction_intensity = glm::vec4(sun_dir, 1.5f);
            auto offset = offsetof(FrameUBO, sun_direction_intensity);
            aligned_frame_buffer_handle.write_field(ctx, bounded_frame_index, sun_direction_intensity, offset);

            constexpr auto rotation_angle = glm::radians(0.5F);
            {
                ZoneScopedNC("Rotate cubes", 0xff0013);
                std::for_each(std::execution::par_unseq, mapped_to_transforms.begin(), mapped_to_transforms.end(),
                              [&random_offset = random_factors, &m = mapped_to_transforms](glm::mat4 &transform) {
                                  // Apply rotation around Y-axis
                                  size_t index = &transform - m.data();
                                  const glm::vec3 &f = random_offset[index];

                                  transform = glm::rotate(transform, rotation_angle * f.y, glm::vec3(0.0f, 1.0f, 0.0f));
                                  transform = glm::rotate(transform, rotation_angle * f.z, glm::vec3(0.0f, 0.0f, 1.0f));
                                  transform = glm::rotate(transform, rotation_angle * f.x, glm::vec3(1.0f, 0.0f, 1.0f));
                              });
            }
            cubes_transform_handle->write_slot(ctx, bounded_frame_index, mapped_to_transforms);
        }

        bindless.repopulate_if_needed(ctx.textures, ctx.samplers);

        auto &fs = frames[bounded_frame_index];

        if (fs.frame_done_value > 0) {
            VkSemaphoreWaitInfo wi{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                   .pNext = nullptr,
                                   .flags = 0,
                                   .semaphoreCount = 1,
                                   .pSemaphores = &tl_graphics.timeline,
                                   .pValues = &fs.frame_done_value};
            vk_check(vkWaitSemaphores(device, &wi, UINT64_MAX));

            if (auto ms = read_timestamp_ms(compute_query_pool[bounded_frame_index]); ms.has_value()) {
                gpu_compute_ms.add_sample(*ms);
            }


            if (auto ms = read_timestamp_ms(graphics_query_pool[bounded_frame_index]); ms.has_value()) {
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

            if (const auto *cqs = ctx.query_pools.get(compute_query_pool[bounded_frame_index])) {
                vkResetQueryPool(device, cqs->pool, 0, cqs->query_count);
            }
            if (const auto *gqs = ctx.query_pools.get(graphics_query_pool[bounded_frame_index])) {
                vkResetQueryPool(device, gqs->pool, 0, gqs->query_count);
            }
            if (const auto *gqs = ctx.query_pools.get(graphics_stats_pool[bounded_frame_index])) {
                vkResetQueryPool(device, gqs->pool, 0, gqs->query_count);
            }
            if (const auto *qs = ctx.query_pools.get(compute_stats_pool[bounded_frame_index])) {
                vkResetQueryPool(device, qs->pool, 0, qs->query_count);
            }
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

        auto predepth_val = submit_stage(
                tl_graphics, device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(tracy_graphics.ctx, cmd, "Predepth");
                    auto depth_handle = (msaa_samples != VK_SAMPLE_COUNT_1_BIT) ? msaa_offscreen_depth_target_handle
                                                                                : offscreen_depth_target_handle;

                    auto &&depth = ctx.textures.get(depth_handle);
                    auto &&[verts, idx] = util::get_mesh_buffers(ctx, cube_predepth_mesh);

                    depth->transition_if_not_initialised(
                            cmd, VK_IMAGE_LAYOUT_GENERAL,
                            {VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT});

                    VkRenderingAttachmentInfo depth_attachment{
                            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                            .imageView = depth->attachment_view,
                            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                            .clearValue = {.depthStencil = {0.0f, 0}},
                    };

                    VkRenderingInfo rendering_info{
                            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                            .renderArea = {.offset = {0, 0}, .extent = {frame_extent.width, frame_extent.height}},
                            .layerCount = 1,
                            .pDepthAttachment = &depth_attachment,
                    };

                    vkCmdBeginRendering(cmd, &rendering_info);
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, predepth_pipeline.pipeline);

                    PredepthPushConstants pc = {
                            .ubo = aligned_frame_buffer_handle.slot_device_address(bounded_frame_index),
                            .transforms = cubes_transform_handle->slot_device_address(bounded_frame_index),
                    };

                    VkViewport vp{
                            .x = 0,
                            .y = static_cast<float>(frame_extent.height),
                            .width = static_cast<float>(frame_extent.width),
                            .height = -static_cast<float>(frame_extent.height),
                            .minDepth = 0.0f,
                            .maxDepth = 1.0f,
                    };

                    VkRect2D sc{.offset = {0, 0}, .extent = {frame_extent.width, frame_extent.height}};
                    vkCmdSetViewport(cmd, 0, 1, &vp);
                    vkCmdSetScissor(cmd, 0, 1, &sc);
                    vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_GREATER_OR_EQUAL);
                    vkCmdSetDepthBounds(cmd, 0.0F, 1.0F);
                    vkCmdSetCullMode(cmd, VK_CULL_MODE_BACK_BIT);
                    vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);
                    vkCmdPushConstants(cmd, predepth_pipeline.layout,
                                       VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);

                    vkCmdBindIndexBuffer(cmd, idx->buffer(), 0, VK_INDEX_TYPE_UINT16);
                    std::array<VkBuffer, 1> buffers = {verts->buffer()};
                    std::array<VkDeviceSize, 1> offsets = {0};
                    vkCmdBindVertexBuffers2(cmd, 0, 1, buffers.data(), offsets.data(), nullptr, nullptr);
                    // draw_mesh(cmd, cube_meshlet_indirect_buf, VkDeviceSize{0}, 1u,
                    //           static_cast<u32>(sizeof(VkDrawMeshTasksIndirectCommandEXT)));
                    vkCmdDrawIndexed(cmd, static_cast<u32>(idx->get_count()), static_cast<u32>(instance_count), 0, 0,
                                     0);

                    vkCmdEndRendering(cmd);
                },
                no_waits);
        fs.timeline_values[stage_index(Stage::Predepth)] = predepth_val;

        const std::array culling_waits{TimelineWait{.value = fs.timeline_values[stage_index(Stage::Predepth)],
                                                    .semaphore = tl_graphics.timeline,
                                                    .stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT}};
        auto light_val = submit_stage(
                tl_compute, device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(tracy_compute.ctx, cmd, "LightCulling");

                    auto &&[cqs, css] = ctx.query_pools.get_multiple(compute_query_pool[bounded_frame_index],
                                                                     compute_stats_pool[bounded_frame_index]);
                    const auto &cqp = cqs->pool;
                    const auto &csp = css->pool;

                    vkCmdResetQueryPool(cmd, cqp, 0, cqs->query_count);
                    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, cqp,
                                        static_cast<u32>(GpuStamp::Begin));
                    vkCmdResetQueryPool(cmd, csp, 0, css->query_count);
                    vkCmdBeginQuery(cmd, csp, 0, 0);

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

                    VkMemoryBarrier2 mem_barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                                 .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                 .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                                                 .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                 .dstAccessMask =
                                                         VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT};

                    VkDependencyInfo dep_info{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                              .memoryBarrierCount = 1,
                                              .pMemoryBarriers = &mem_barrier};

                    // ---------------------------------------------------------------------
                    // Clear required buffers
                    // ---------------------------------------------------------------------
                    fill_zeros(cmd, ctx.buffers, flags_handle, prefix_handle, compact_lights_handle,
                               culled_light_count_handle);

                    vkCmdPipelineBarrier2(cmd, &dep_info);

                    // ---------------------------------------------------------------------
                    // Pass 1: flags
                    // ---------------------------------------------------------------------
                    const u32 gc = (light_count + THREADS_PER_GROUP - 1) / THREADS_PER_GROUP;

                    bind_and_dispatch(flags_pipeline, gc);
                    vkCmdPipelineBarrier2(cmd, &dep_info);

                    // ---------------------------------------------------------------------
                    // Pass 2: scan + compact (atomic reservation)
                    // ---------------------------------------------------------------------
                    bind_and_dispatch(compact_pipeline, gc);

                    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, cqp,
                                        static_cast<u32>(GpuStamp::End));
                    vkCmdEndQuery(cmd, csp, 0);

                    TRACY_GPU_COLLECT(tracy_compute.ctx, cmd);
                },
                SubmitSynchronisation{.timeline_waits = culling_waits});

        fs.timeline_values[stage_index(Stage::LightCulling)] = light_val;

        const std::array gbuffer_waits{
                TimelineWait{
                        .value = fs.timeline_values[stage_index(Stage::LightCulling)],
                        .semaphore = tl_compute.timeline,
                        .stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                },
        };

        auto gbuffer_val = submit_stage(
                tl_graphics, device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(tracy_graphics.ctx, cmd, "GBuffer");
                    auto &&[graphics_perf_query, graphics_stats] = ctx.query_pools.get_multiple(
                            graphics_query_pool[bounded_frame_index], graphics_stats_pool[bounded_frame_index]);
                    auto &&[verts, idx] = util::get_mesh_buffers(ctx, cube_mesh);

                    const bool msaa_enabled = (msaa_samples != VK_SAMPLE_COUNT_1_BIT);

                    auto &&resolve = ctx.textures.get(offscreen_target_handle);

                    auto *color = msaa_enabled ? ctx.textures.get(msaa_offscreen_target_handle) : resolve;
                    auto *depth = msaa_enabled ? ctx.textures.get(msaa_offscreen_depth_target_handle)
                                               : ctx.textures.get(offscreen_depth_target_handle);

                    const VkQueryPool &graphics_perf_pool = graphics_perf_query->pool;
                    const VkQueryPool &graphics_pool_for_stats = graphics_stats->pool;

                    vkCmdResetQueryPool(cmd, graphics_perf_pool, 0, graphics_perf_query->query_count);
                    vkCmdResetQueryPool(cmd, graphics_pool_for_stats, 0, graphics_stats->query_count);
                    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, graphics_perf_pool, 0);
                    vkCmdBeginQuery(cmd, graphics_pool_for_stats, 0, 0);

                    resolve->transition_if_not_initialised(
                            cmd, VK_IMAGE_LAYOUT_GENERAL,
                            {VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT});
                    color->transition_if_not_initialised(
                            cmd, VK_IMAGE_LAYOUT_GENERAL,
                            {VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT});
                    depth->transition_if_not_initialised(
                            cmd, VK_IMAGE_LAYOUT_GENERAL,
                            {VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT});

                    VkRenderingAttachmentInfo color_attachment{
                            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                            .resolveMode = VK_RESOLVE_MODE_NONE,
                            .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
                            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                            .clearValue = {.color = {0.0f, 0.0f, 0.0f, 1.0f}},
                    };

                    color_attachment.imageView = color->attachment_view;
                    color_attachment.storeOp =
                            msaa_enabled ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;

                    if (msaa_enabled) {
                        color_attachment.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
                        color_attachment.resolveImageView = resolve->attachment_view;
                        color_attachment.resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    }

                    VkRenderingAttachmentInfo depth_attachment{
                            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                            .imageView = depth->attachment_view,
                            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                            .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                    };

                    VkRenderingInfo rendering_info{
                            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                            .renderArea = {.offset = {0, 0}, .extent = {frame_extent.width, frame_extent.height}},
                            .layerCount = 1,
                            .colorAttachmentCount = 1,
                            .pColorAttachments = &color_attachment,
                            .pDepthAttachment = &depth_attachment,
                    };

                    vkCmdBeginRendering(cmd, &rendering_info);

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, point_light_pipeline.pipeline);

                    const RenderingPushConstants pc{
                            .ubo = aligned_frame_buffer_handle.slot_device_address(bounded_frame_index),
                            .transforms = cubes_transform_handle->slot_device_address(bounded_frame_index),
                    };

                    VkViewport vp{
                            .x = 0,
                            .y = static_cast<float>(frame_extent.height),
                            .width = static_cast<float>(frame_extent.width),
                            .height = -static_cast<float>(frame_extent.height),
                            .minDepth = 0.0f,
                            .maxDepth = 1.0f,
                    };

                    VkRect2D sc{.offset = {0, 0}, .extent = {frame_extent.width, frame_extent.height}};

                    vkCmdSetViewport(cmd, 0, 1, &vp);
                    vkCmdSetScissor(cmd, 0, 1, &sc);
                    vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_EQUAL);
                    vkCmdSetCullMode(cmd, VK_CULL_MODE_BACK_BIT);
                    vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);
                    vkCmdSetDepthBounds(cmd, 0.0F, 1.0F);
                    vkCmdPushConstants(cmd, point_light_pipeline.layout,
                                       VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
                    vkCmdBindIndexBuffer(cmd, idx->buffer(), 0, VK_INDEX_TYPE_UINT16);
                    std::array<VkBuffer, 1> buffers = {verts->buffer()};
                    std::array<VkDeviceSize, 1> offsets = {0};
                    vkCmdBindVertexBuffers2(cmd, 0, 1, buffers.data(), offsets.data(), nullptr, nullptr);
                    // draw_mesh(cmd, cube_meshlet_indirect_buf, VkDeviceSize{0}, 1u,
                    //           static_cast<u32>(sizeof(VkDrawMeshTasksIndirectCommandEXT)));
                    vkCmdDrawIndexed(cmd, static_cast<u32>(idx->get_count()), static_cast<u32>(instance_count), 0, 0,
                                     0);

                    vkCmdEndRendering(cmd);

                    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, graphics_perf_pool, 1);
                    vkCmdEndQuery(cmd, graphics_pool_for_stats, 0);
                },
                SubmitSynchronisation{.timeline_waits = gbuffer_waits});
        fs.timeline_values[stage_index(Stage::GBuffer)] = gbuffer_val;

        const std::array tonemap_waits{
                TimelineWait{
                        .value = fs.timeline_values[stage_index(Stage::GBuffer)],
                        .semaphore = tl_graphics.timeline,
                        .stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                },
        };

        auto tonemap_val = submit_stage(
                tl_graphics, device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(tracy_graphics.ctx, cmd, "Tonemapping");
                    auto &&hdr = ctx.textures.get(offscreen_target_handle);
                    auto &&ldr = ctx.textures.get(tonemapped_target_handle);

                    // Transition HDR for sampling
                    hdr->transition_if_not_initialised(
                            cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            {VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT});

                    // Transition LDR for rendering
                    ldr->transition_if_not_initialised(
                            cmd, VK_IMAGE_LAYOUT_GENERAL,
                            {VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT});

                    VkRenderingAttachmentInfo color_attachment{
                            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                            .imageView = ldr->sampled_view,
                            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                            .clearValue = {.color = {0, 0, 0, 1}},
                    };

                    VkRenderingInfo ri{
                            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                            .renderArea = {.offset = {0, 0}, .extent = {frame_extent.width, frame_extent.height}},
                            .layerCount = 1,
                            .colorAttachmentCount = 1,
                            .pColorAttachments = &color_attachment,
                    };

                    vkCmdBeginRendering(cmd, &ri);

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tonemap_pipeline.pipeline);

                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tonemap_pipeline.layout, 0, 1,
                                            &bindless.set, 0, nullptr);

                    float exposure = 1.0f;
                    TonemapPushConstants pc{
                            .exposure = exposure,
                            .image_index = offscreen_target_handle.index(),
                            .sampler_index = linear_clamp_sampler_handle.index(),
                    };


                    VkViewport vp{
                            .x = 0,
                            .y = static_cast<float>(frame_extent.height),
                            .width = static_cast<float>(frame_extent.width),
                            .height = -static_cast<float>(frame_extent.height),
                            .minDepth = 0.0f,
                            .maxDepth = 1.0f,
                    };

                    VkRect2D sc{.offset = {0, 0}, .extent = {frame_extent.width, frame_extent.height}};
                    vkCmdSetViewport(cmd, 0, 1, &vp);
                    vkCmdSetScissor(cmd, 0, 1, &sc);

                    vkCmdPushConstants(cmd, tonemap_pipeline.layout,
                                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

                    vkCmdDraw(cmd, 3, 1, 0, 0); // fullscreen triangle

                    vkCmdEndRendering(cmd);
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
                .stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        }};

        const std::array present_binary_signals{frame_sync.render_finished};

        auto swapchain_val = submit_stage(
                tl_graphics, device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(tracy_graphics.ctx, cmd, "CopyToSwapchain");
                    auto &&tonemapped = ctx.textures.get(tonemapped_target_handle);

                    const auto dst_image = swapchain.image(swap_image_index);
                    const auto src_image = tonemapped->image;

                    const std::array barriers{VkImageMemoryBarrier2{
                                                      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                                      .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                                      .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                                      .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                      .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                                                      .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                                      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                      .image = src_image,
                                                      .subresourceRange =
                                                              VkImageSubresourceRange{
                                                                      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                                      .baseMipLevel = 0,
                                                                      .levelCount = 1,
                                                                      .baseArrayLayer = 0,
                                                                      .layerCount = 1,
                                                              },
                                              },
                                              VkImageMemoryBarrier2{
                                                      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                                      .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                                                      .srcAccessMask = VK_ACCESS_2_NONE,
                                                      .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                      .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                                      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                      .image = dst_image,
                                                      .subresourceRange =
                                                              VkImageSubresourceRange{
                                                                      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                                      .baseMipLevel = 0,
                                                                      .levelCount = 1,
                                                                      .baseArrayLayer = 0,
                                                                      .layerCount = 1,
                                                              },
                                              }};

                    VkDependencyInfo dep_info{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                              .imageMemoryBarrierCount = static_cast<u32>(barriers.size()),
                                              .pImageMemoryBarriers = barriers.data()};

                    vkCmdPipelineBarrier2(cmd, &dep_info);

                    VkImageCopy region{
                            .srcSubresource =
                                    VkImageSubresourceLayers{
                                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                            .mipLevel = 0,
                                            .baseArrayLayer = 0,
                                            .layerCount = 1,
                                    },
                            .srcOffset = VkOffset3D{0, 0, 0},
                            .dstSubresource =
                                    VkImageSubresourceLayers{
                                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                            .mipLevel = 0,
                                            .baseArrayLayer = 0,
                                            .layerCount = 1,
                                    },
                            .dstOffset = VkOffset3D{0, 0, 0},
                            .extent = VkExtent3D{frame_extent.width, frame_extent.height, 1},
                    };

                    vkCmdCopyImage(cmd, src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst_image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

                    const std::array end_barriers{
                            VkImageMemoryBarrier2{
                                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                    .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                    .srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                                    .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                    .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                    .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .image = src_image,
                                    .subresourceRange =
                                            VkImageSubresourceRange{
                                                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                    .baseMipLevel = 0,
                                                    .levelCount = 1,
                                                    .baseArrayLayer = 0,
                                                    .layerCount = 1,
                                            },
                            },
                            VkImageMemoryBarrier2{
                                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                    .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                    .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                    .dstStageMask = VK_PIPELINE_STAGE_2_NONE,
                                    .dstAccessMask = VK_ACCESS_2_NONE,
                                    .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .image = dst_image,
                                    .subresourceRange =
                                            VkImageSubresourceRange{
                                                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                    .baseMipLevel = 0,
                                                    .levelCount = 1,
                                                    .baseArrayLayer = 0,
                                                    .layerCount = 1,
                                            },
                            },
                    };

                    VkDependencyInfo end_dep_info{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                                  .imageMemoryBarrierCount = static_cast<u32>(end_barriers.size()),
                                                  .pImageMemoryBarriers = end_barriers.data()};

                    vkCmdPipelineBarrier2(cmd, &end_dep_info);
                    TRACY_GPU_COLLECT(tracy_graphics.ctx, cmd);
                },
                SubmitSynchronisation{
                        .timeline_waits = present_timeline_waits,
                        .binary_waits = present_binary_waits,
                        .binary_signals = present_binary_signals,
                });

        fs.frame_done_value = swapchain_val;

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

        time_total += ms / 1000.0;
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

    const auto &&[oth, ph] = ctx.textures.get_multiple(offscreen_target_handle, perlin_handle);
    {
        ZoneScopedNC("batch_write_images", 0xFF00AA);
        std::array requests{image_operations::ImageWriteRequest{oth, "output.bmp"},
                            image_operations::ImageWriteRequest{ph, "perlin.bmp"}};
        image_operations::write_batch_to_disk(allocator, requests);
    }

    pipeline_cache.reset();
    ctx.clear_all();


    watcher.reset();
    listeners.clear();

    ctx.destroy_queue.retire(UINT64_MAX);

    tracy_compute.shutdown();
    tracy_graphics.shutdown();

    destruction::pipeline(device, flags_pipeline, compact_pipeline, predepth_pipeline, tonemap_pipeline,
                          point_light_pipeline);
    destruction::global_command_context(command_context);
    destruction::bindless_set(device, bindless);
    destruction::timeline_compute(device, tl_graphics);
    destruction::timeline_compute(device, tl_compute);
    destruction::allocator(allocator);
    swapchain.destroy();
    destruction::wsi(instance.instance, surface, window);
    vkDeviceWaitIdle(device);
    destruction::device(device);
    destruction::instance(instance);

    return 0;
}


auto main(int argc, char **argv) -> int {
    execute(argc, argv);

    info("Bindless headless setup and teardown completed successfully.");
    return 0;
}
