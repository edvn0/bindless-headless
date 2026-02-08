#include "app/app.hxx"

#include <3PP/stb_image.h>
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
#include <imgui.h>
#include <iostream>
#include <random>
#include <ranges>
#include <thread>
#include <unordered_map>

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
#include "app/ui.hxx"


// forward declare internal pieces
struct AppGpuState;
struct AppPipelines;
struct AppFrameResources;

// ------------------------------------------------------------
// App-side input/events
// ------------------------------------------------------------

struct AppState {
    bool resized{false};

    glm::vec2 last_mouse{0.0f, 0.0f};
    bool mouse_inited{false};
    EventSystem event_system{};

    CameraInput cam_in{};
    EditorCamera cam{};
};

// ------------------------------------------------------------
// Refactor buckets
// ------------------------------------------------------------

struct AppGpuState {
    // Lifetime order matters. Keep “parents” above “children”.

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
    GlobalCommandContext command_context{};

    VmaAllocator allocator{VK_NULL_HANDLE};

    ComputeTimeline tl_compute{};
    GraphicsTimeline tl_graphics{};
    TransferTimeline tl_transfer{};

    BindlessSet bindless{};

    RenderContext ctx{};

    std::unique_ptr<Compiler> compiler{};

    VkSampleCountFlagBits msaa_samples{VK_SAMPLE_COUNT_1_BIT};
};

struct AppPipelines {
    ShaderHandle fullscreen_vs{};

    PipelineHandle flags_pipeline{};
    PipelineHandle compact_pipeline{};
    PipelineHandle cube_rotation_pipeline{};
    PipelineHandle gbuffer_pipeline_mrt{};
    PipelineHandle gbuffer_pipeline_lighting{};
    PipelineHandle predepth_pipeline{};
    PipelineHandle predepth_alpha_pipeline{};
    PipelineHandle tonemap_pipeline{};
    PipelineHandle cluster_build_groups_pipeline{};
    PipelineHandle present_pipeline{};

    std::array<QueryPoolHandle, frames_in_flight> compute_query_pool{};
    std::array<QueryPoolHandle, frames_in_flight> graphics_query_pool{};
    std::array<QueryPoolHandle, frames_in_flight> graphics_stats_pool{};
    std::array<QueryPoolHandle, frames_in_flight> compute_stats_pool{};

    SamplerHandle linear_repeat{};
    SamplerHandle linear_clamp{};
    SamplerHandle noise_sampler{};
};

struct AppResources {
    std::array<FrameState, frames_in_flight> frames{};

    LoadedObj mesh{};

    std::vector<PointLight> all_point_lights{};
    std::vector<PointLight> all_point_lights_zero{};
    u32 light_count{0};

    BufferHandle point_lights_base{};
    AlignedRingBuffer<PointLight> point_lights_ring{};
    BufferHandle culled_light_count{};

    static constexpr u32 mesh_count = 1;
    AlignedRingBuffer<glm::mat4x3> transforms_ring{};
    u32 instance_count{0};

    BufferHandle flags{};
    BufferHandle prefix{};
    BufferHandle compact_lights{};

    ClusterConfig clustering_config{};
    u32 max_light_indices{0};

    BufferHandle cluster_counts{};
    BufferHandle visibility{};
    BufferHandle clusters{};
    BufferHandle cluster_counters{};
    BufferHandle cluster_light_indices{};
    BufferHandle global_index_count{};

    AlignedRingBuffer<FrameUBO> frame_ubo_ring{};

    TextureHandle gbuffer0{};
    TextureHandle gbuffer1{};
    TextureHandle gbuffer2{};
    TextureHandle debug_culling{};
    TextureHandle lit_hdr{};
    TextureHandle depth{};
    TextureHandle tonemapped{};

    TextureHandle perlin_noise{};

    static constexpr u32 max_draws_per_frame = 100000U;
    AlignedRingBuffer<VkDrawIndexedIndirectCommand> indirect_ring{};
    AlignedRingBuffer<u32> draw_material_id_ring{};

    struct FrameDrawStream {
        FrameIndirectWriter writer{};
        auto begin_frame() -> void { writer.cursor = 0; }
    } draw_stream{};
};

struct AppUi {
    AppState app_state{};
    std::unique_ptr<ImGuiRenderer> gui{};
    std::unique_ptr<efsw::FileWatcher, Deleter> watcher{};
    std::unordered_map<std::string, std::unique_ptr<efsw::FileWatchListener, Deleter>> listeners{};

    u64 frame_index{};
    std::chrono::high_resolution_clock::time_point last_frame_time{};
    double dt{0.0};
    double total_time{0.0};

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
    };

    ClusterDebugMode debug_mode{ClusterDebugMode::None};

    // graphs
    PerformanceGraph<8, 120> gpu_frame_graph{};
    bool graphs_initialized{false};
};

extern auto ImGui_KeyToImGuiKey(int key) -> ImGuiKey;


// ------------------------------------------------------------
// BindlessApp private helpers (declared in-class in app.hxx or here as local lambdas)
// ------------------------------------------------------------

namespace {
    auto fill_zeros(VkCommandBuffer cmd, auto &buffers_ctx, auto &&...buffer_handles) {
        (vkCmdFillBuffer(cmd, buffers_ctx.get(buffer_handles)->buffer(), 0, VK_WHOLE_SIZE, 0), ...);
    }

    auto set_window_callbacks(GLFWwindow *window, AppUi &ui) -> void {
        glfwSetWindowUserPointer(window, &ui.app_state);

        glfwSetKeyCallback(window, [](GLFWwindow *w, int key, int scancode, int action, int mods) {
            auto &app = *static_cast<AppState *>(glfwGetWindowUserPointer(w));

            auto &io = ImGui::GetIO();
            if (action == GLFW_PRESS) {
                io.AddKeyEvent(ImGui_KeyToImGuiKey(key), true);
            } else if (action == GLFW_RELEASE) {
                io.AddKeyEvent(ImGui_KeyToImGuiKey(key), false);
            }

            if (!io.WantCaptureKeyboard) {
                if (action == GLFW_PRESS) {
                    auto event = std::make_unique<KeyPressedEvent>();
                    event->key = key;
                    event->scancode = scancode;
                    event->mods = mods;
                    app.event_system.push_event(std::move(event));
                } else if (action == GLFW_RELEASE) {
                    auto event = std::make_unique<KeyReleasedEvent>();
                    event->key = key;
                    event->scancode = scancode;
                    event->mods = mods;
                    app.event_system.push_event(std::move(event));
                }
            }
        });

        glfwSetCharCallback(window, [](GLFWwindow *w, unsigned int c) {
            auto &app = *static_cast<AppState *>(glfwGetWindowUserPointer(w));

            auto &io = ImGui::GetIO();
            io.AddInputCharacter(c);

            if (!io.WantCaptureKeyboard) {
                auto event = std::make_unique<CharInputEvent>();
                event->codepoint = c;
                app.event_system.push_event(std::move(event));
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

        glfwSetMouseButtonCallback(window, [](GLFWwindow *w, int button, int action, int mods) {
            auto &app = *static_cast<AppState *>(glfwGetWindowUserPointer(w));
            auto &io = ImGui::GetIO();

            if (action == GLFW_PRESS && button >= 0 && button < ImGuiMouseButton_COUNT) {
                io.AddMouseButtonEvent(button, true);
            } else if (action == GLFW_RELEASE && button >= 0 && button < ImGuiMouseButton_COUNT) {
                io.AddMouseButtonEvent(button, false);
            }

            if (!io.WantCaptureMouse) {
                if (action == GLFW_PRESS) {
                    auto event = std::make_unique<MouseButtonPressedEvent>();
                    event->button = button;
                    event->mods = mods;
                    app.event_system.push_event(std::move(event));
                } else if (action == GLFW_RELEASE) {
                    auto event = std::make_unique<MouseButtonReleasedEvent>();
                    event->button = button;
                    event->mods = mods;
                    app.event_system.push_event(std::move(event));
                }
            }
        });

        glfwSetCursorPosCallback(window, [](GLFWwindow *w, double x, double y) {
            auto &app = *static_cast<AppState *>(glfwGetWindowUserPointer(w));

            auto &io = ImGui::GetIO();
            io.AddMousePosEvent(static_cast<float>(x), static_cast<float>(y));

            const glm::vec2 pos{static_cast<float>(x), static_cast<float>(y)};
            glm::vec2 delta{0.0f};

            if (!app.mouse_inited) {
                app.last_mouse = pos;
                app.mouse_inited = true;
            } else {
                delta = pos - app.last_mouse;
                app.last_mouse = pos;
            }

            if (!io.WantCaptureMouse) {
                auto event = std::make_unique<CursorMovedEvent>();
                event->position = pos;
                event->delta = delta;
                app.event_system.push_event(std::move(event));
            }
        });

        glfwSetScrollCallback(window, [](GLFWwindow *w, double xoff, double yoff) {
            auto &app = *static_cast<AppState *>(glfwGetWindowUserPointer(w));

            auto &io = ImGui::GetIO();
            io.AddMouseWheelEvent(static_cast<float>(xoff), static_cast<float>(yoff));

            if (!io.WantCaptureMouse) {
                auto event = std::make_unique<ScrollEvent>();
                event->x_offset = static_cast<float>(xoff);
                event->y_offset = static_cast<float>(yoff);
                app.event_system.push_event(std::move(event));
            }
        });
    }

    auto wire_event_dispatch(GLFWwindow *window, AppUi &ui) -> void {
        ui.app_state.event_system.set_event_callback([&](Event &e) {
            auto &io = ImGui::GetIO();
            EventDispatcher dispatcher(e);

            dispatcher.dispatch<KeyPressedEvent>([&](KeyPressedEvent &event) {
                if (event.key == GLFW_KEY_ESCAPE) {
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                    return true;
                }

                // Debug mode hotkeys
                if (event.key == GLFW_KEY_F1) {
                    ui.debug_mode = AppUi::ClusterDebugMode::ClusterGrid;
                    return true;
                } else if (event.key == GLFW_KEY_F2) {
                    ui.debug_mode = AppUi::ClusterDebugMode::LightCount;
                    return true;
                } else if (event.key == GLFW_KEY_F3) {
                    ui.debug_mode = AppUi::ClusterDebugMode::LightDensity;
                    return true;
                } else if (event.key == GLFW_KEY_F4) {
                    ui.debug_mode = AppUi::ClusterDebugMode::DepthSlices;
                    return true;
                } else if (event.key == GLFW_KEY_F5) {
                    ui.debug_mode = AppUi::ClusterDebugMode::LightHeatmap;
                    return true;
                } else if (event.key == GLFW_KEY_F6) {
                    ui.debug_mode = AppUi::ClusterDebugMode::FirstLight;
                    return true;
                } else if (event.key == GLFW_KEY_F7) {
                    ui.debug_mode = AppUi::ClusterDebugMode::ClusterOccupancy;
                    return true;
                } else if (event.key == GLFW_KEY_F8) {
                    ui.debug_mode = AppUi::ClusterDebugMode::None;
                    return true;
                }

                return false;
            });

            dispatcher.dispatch<MouseButtonPressedEvent>([&](MouseButtonPressedEvent &event) {
                if (event.button == GLFW_MOUSE_BUTTON_LEFT) {
                    ui.app_state.cam_in.lmb = true;
                    return true;
                } else if (event.button == GLFW_MOUSE_BUTTON_MIDDLE) {
                    ui.app_state.cam_in.mmb = true;
                    return true;
                } else if (event.button == GLFW_MOUSE_BUTTON_RIGHT) {
                    ui.app_state.cam_in.rmb = true;
                    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                    return true;
                }
                return false;
            });

            dispatcher.dispatch<MouseButtonReleasedEvent>([&](MouseButtonReleasedEvent &event) {
                if (io.WantCaptureMouse) {
                    return false;
                }

                if (event.button == GLFW_MOUSE_BUTTON_LEFT) {
                    ui.app_state.cam_in.lmb = false;
                    return true;
                } else if (event.button == GLFW_MOUSE_BUTTON_MIDDLE) {
                    ui.app_state.cam_in.mmb = false;
                    return true;
                } else if (event.button == GLFW_MOUSE_BUTTON_RIGHT) {
                    ui.app_state.cam_in.rmb = false;
                    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                    return true;
                }
                return false;
            });

            dispatcher.dispatch<CursorMovedEvent>([&](CursorMovedEvent &event) {
                if (io.WantCaptureMouse) {
                    return false;
                }
                ui.app_state.cam_in.mouse_delta += event.delta;
                return true;
            });

            dispatcher.dispatch<ScrollEvent>([&](ScrollEvent &event) {
                if (io.WantCaptureMouse) {
                    return false;
                }
                ui.app_state.cam_in.scroll_delta += event.y_offset;
                return true;
            });
        });
    }

} // namespace

auto BindlessApp::run(CLIOptions &opts, InstanceWithDebug &instance) -> tl::expected<int, Error> {

    AppGpuState gpu{};
    AppPipelines pipes{};
    AppResources res{};
    AppUi ui{};

    gpu.opts = &opts;
    gpu.instance = &instance;

    gpu.compiler = std::make_unique<Compiler>();

    {
        auto could_choose = pick_physical_device(instance.instance);
        if (!could_choose) {
            return tl::make_unexpected(
                    Error::make_error(Error::Type::DeviceSelectionError, "Failed to choose physical device"));
        }

        auto &&[physical, gfx_i, comp_i, xfer_i] = *could_choose;
        gpu.physical_device = physical;
        gpu.graphics_index = gfx_i;
        gpu.compute_index = comp_i;
        gpu.transfer_index = xfer_i;

        auto &&[device, gfx_q, comp_q, xfer_q, enabled] =
                create_device(gpu.physical_device, gpu.graphics_index, gpu.compute_index, gpu.transfer_index);

        gpu.device = device;
        gpu.graphics_queue = gfx_q;
        gpu.compute_queue = comp_q;
        gpu.transfer_queue = xfer_q;
        gpu.enabled_features = std::move(enabled);
    }

    gpu.tracy_graphics.init_calibrated(instance, gpu.physical_device, gpu.device, "Graphics Queue");
    gpu.tracy_compute.init_calibrated(instance, gpu.physical_device, gpu.device, "Compute Queue");

    {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

        gpu.window = glfwCreateWindow(static_cast<i32>(opts.width), static_cast<i32>(opts.height), "Bindless", nullptr,
                                      nullptr);
        if (!gpu.window) {
            error("Could not create window");
            return 1;
        }

        vk_check(glfwCreateWindowSurface(instance.instance, gpu.window, nullptr, &gpu.surface));
    }

    {
        auto maybe_swapchain = Swapchain::create(SwapchainCreateInfo{
                .physical_device = gpu.physical_device,
                .device = gpu.device,
                .surface = gpu.surface,
                .graphics_family = gpu.graphics_index,
                .extent = VkExtent2D{opts.width, opts.height},
                .vsync = opts.vsync,
        });
        if (!maybe_swapchain) {
            return 1;
        }
        gpu.swapchain = std::move(maybe_swapchain.value());
    }

    // --- Command context + allocator + timelines ---
    gpu.command_context = create_global_cmd_context(gpu.device, gpu.graphics_queue, gpu.graphics_index);
    gpu.allocator = create_allocator(instance.instance, gpu.physical_device, gpu.device);

    gpu.tl_compute = create_compute_timeline(gpu.device, gpu.compute_queue, gpu.compute_index);
    gpu.tl_graphics = create_graphics_timeline(gpu.device, gpu.graphics_queue, gpu.graphics_index);
    gpu.tl_transfer = create_transfer_timeline(gpu.device, gpu.transfer_queue, gpu.transfer_index);

    gpu.bindless.init(gpu.device, query_bindless_caps(gpu.physical_device), 8u, 8u, 8u, 0u);
    gpu.bindless.grow_if_needed(300u, 40u, 32u, 8u);

    {
        const VkSampleCountFlagBits requested = msaa_from_cli(opts.msaa);
        gpu.msaa_samples = clamp_msaa_samples(gpu.physical_device, requested);
        info("MSAA requested: {}, Engine supplied: {}", static_cast<u32>(requested),
             static_cast<u32>(gpu.msaa_samples));

        gpu.ctx = RenderContext{
                .allocator = gpu.allocator,
                .bindless_set = &gpu.bindless,
                .pipeline_cache = std::make_unique<PipelineCache>(gpu.device, opts.pipeline_cache_dir),
                .queues =
                        {
                                .graphics = {gpu.graphics_queue, gpu.graphics_index},
                                .compute = {gpu.compute_queue, gpu.compute_index},
                                .transfer = {gpu.transfer_queue, gpu.transfer_index},
                        },
        };

        pipes.fullscreen_vs = gpu.ctx.shaders.get_handle(get_or_create_fullscreen_vs(gpu.ctx));
    }

    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(gpu.physical_device, &props);
        const auto timestamp_period_ns = static_cast<double>(props.limits.timestampPeriod);

        const VkQueryPoolCreateFlags reset_flags = gpu.enabled_features.contains(VK_KHR_MAINTENANCE_9_EXTENSION_NAME)
                                                           ? VK_QUERY_POOL_CREATE_RESET_BIT_KHR
                                                           : 0;

        for (u32 fi = 0; fi < frames_in_flight; ++fi) {
            auto qpci = create_info<VkQueryPoolCreateInfo>();
            qpci.flags = reset_flags;
            qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qpci.queryCount = compute_query_count;

            VkQueryPool qpc = VK_NULL_HANDLE;
            vk_check(vkCreateQueryPool(gpu.device, &qpci, nullptr, &qpc));

            pipes.compute_query_pool[fi] = gpu.ctx.create_query_pool(QueryPoolState{
                    .pool = qpc, .query_count = compute_query_count, .timestamp_period_ns = timestamp_period_ns});
            set_debug_name(gpu.device, VK_OBJECT_TYPE_QUERY_POOL, qpc,
                           std::format("compute_timestamp_query_pool_frame_{}", fi));

            qpci.queryCount = graphics_query_count;

            VkQueryPool qpg = VK_NULL_HANDLE;
            vk_check(vkCreateQueryPool(gpu.device, &qpci, nullptr, &qpg));

            pipes.graphics_query_pool[fi] = gpu.ctx.create_query_pool(QueryPoolState{
                    .pool = qpg, .query_count = graphics_query_count, .timestamp_period_ns = timestamp_period_ns});
            set_debug_name(gpu.device, VK_OBJECT_TYPE_QUERY_POOL, qpg,
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
            vk_check(vkCreateQueryPool(gpu.device, &stats_qpci, nullptr, &stats_pool));
            pipes.graphics_stats_pool[fi] = gpu.ctx.create_query_pool(QueryPoolState{
                    .pool = stats_pool, .query_count = stats_graphics_count, .timestamp_period_ns = 0.0});
            set_debug_name(gpu.device, VK_OBJECT_TYPE_QUERY_POOL, stats_pool,
                           std::format("graphics_stats_query_pool_frame_{}", fi));

            VkQueryPoolCreateInfo compute_stats_qpci{
                    .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = reset_flags,
                    .queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS,
                    .queryCount = stats_compute_count,
                    .pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT,
            };

            VkQueryPool compute_stats = VK_NULL_HANDLE;
            vk_check(vkCreateQueryPool(gpu.device, &compute_stats_qpci, nullptr, &compute_stats));
            pipes.compute_stats_pool[fi] = gpu.ctx.create_query_pool(QueryPoolState{
                    .pool = compute_stats, .query_count = stats_compute_count, .timestamp_period_ns = 0.0});
            set_debug_name(gpu.device, VK_OBJECT_TYPE_QUERY_POOL, compute_stats,
                           std::format("compute_stats_query_pool_frame_{}", fi));

            if (reset_flags == 0) {
                vkResetQueryPool(gpu.device, qpc, 0, compute_query_count);
                vkResetQueryPool(gpu.device, qpg, 0, graphics_query_count);
                vkResetQueryPool(gpu.device, stats_pool, 0, stats_graphics_count);
                vkResetQueryPool(gpu.device, compute_stats, 0, stats_compute_count);
            }
        }
    }

    // --- Default textures (white/black/flat-normal) ---
    {
        std::array<u8, 4> white{255, 255, 255, 255};
        std::array<u8, 4> black{0, 0, 0, 255};
        std::array<u8, 4> flat_normal{128, 128, 255, 255};

        auto white_handle = gpu.ctx.create_texture(
                create_image_from_span_v2(gpu.allocator, gpu.command_context, 1, 1, VK_FORMAT_R8G8B8A8_UNORM,
                                          std::as_bytes(std::span(white)), "white-texture"));
        auto black_handle = gpu.ctx.create_texture(
                create_image_from_span_v2(gpu.allocator, gpu.command_context, 1, 1, VK_FORMAT_R8G8B8A8_UNORM,
                                          std::as_bytes(std::span(black)), "black-texture"));
        auto flat_normal_handle = gpu.ctx.create_texture(
                create_image_from_span_v2(gpu.allocator, gpu.command_context, 1, 1, VK_FORMAT_R8G8B8A8_UNORM,
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

    {
        auto noise = generate_perlin(2048, 2048);
        res.perlin_noise =
                gpu.ctx.create_texture(create_image_from_span_v2(gpu.allocator, gpu.command_context, 2048u, 2048u,
                                                                 VK_FORMAT_R8_UNORM, std::span{noise}, "perlin_noise"));
    }

    {
        {
            auto ci = create_info<VkSamplerCreateInfo>();
            ci.magFilter = VK_FILTER_LINEAR;
            ci.minFilter = VK_FILTER_LINEAR;
            ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            ci.maxAnisotropy = 16.0f;
            ci.anisotropyEnable = VK_TRUE;
            ci.maxLod = VK_LOD_CLAMP_NONE;
            ci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

            pipes.linear_repeat = gpu.ctx.create_sampler(ci, "linear_repeat");
            info("Linear Repeat Sampler Index: {}", pipes.linear_repeat.index());
        }

        {
            auto ci = create_info<VkSamplerCreateInfo>();
            ci.magFilter = VK_FILTER_LINEAR;
            ci.minFilter = VK_FILTER_LINEAR;
            ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            ci.compareOp = VK_COMPARE_OP_ALWAYS;
            ci.maxLod = VK_LOD_CLAMP_NONE;
            ci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

            pipes.linear_clamp = gpu.ctx.create_sampler(ci, "linear_clamp");
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

            gpu.ctx.create_sampler(create_sampler(gpu.allocator, ci, "noise_sampler"));
            // If create_sampler returns a handle in your codebase, store it in pipes.noise_sampler.
        }
    }

    gpu.bindless.repopulate_if_needed(gpu.ctx.textures, gpu.ctx.samplers);

    // --- Load meshes ---
    TRY_PROPAGATE(loaded_mesh, load_obj(gpu.ctx, gpu.command_context, "assets/meshes/Sponza-master/sponza.obj"),
                  "Failed to load cube mesh");
    res.mesh = std::move(loaded_mesh);

    // --- Lights + buffers ---
    {
        res.all_point_lights = std::vector<PointLight>(opts.light_count);
        res.all_point_lights_zero = std::vector<PointLight>(opts.light_count);
        res.light_count = static_cast<u32>(res.all_point_lights.size());

        const auto mesh_aabb = res.mesh.mesh_aabb;
        info("Mesh AABB: min({}, {}, {}) max({}, {}, {})", mesh_aabb.min.x, mesh_aabb.min.y, mesh_aabb.min.z,
             mesh_aabb.max.x, mesh_aabb.max.y, mesh_aabb.max.z);

        spawn_lights_in_aabb(mesh_aabb, res.all_point_lights);

        res.point_lights_base =
                gpu.ctx.buffers.create(Buffer::from_slice<PointLight>(gpu.allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                                      res.all_point_lights, "base_static_point_lights")
                                               .value());

        auto ring = AlignedRingBuffer<PointLight>::create(gpu.ctx, res.light_count, VkBufferUsageFlags{},
                                                          "point_lights_ring");
        res.point_lights_ring = std::move(ring.value());
        res.point_lights_ring.write_all_slots(gpu.ctx, res.all_point_lights);

        res.culled_light_count = gpu.ctx.buffers.create(
                Buffer::from_value<u32>(gpu.allocator,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0u,
                                        "culled_point_light_count")
                        .value());

        auto transforms =
                AlignedRingBuffer<glm::mat4x3>::create(gpu.ctx, res.mesh_count, VkBufferUsageFlags{}, "transforms");
        res.transforms_ring = std::move(*transforms);
        res.transforms_ring.write_all_slots(gpu.ctx, glm::identity<glm::mat4x3>());

        res.instance_count = static_cast<u32>(res.mesh_count);

        std::vector zeros_lights(res.light_count, 0u);
        res.flags = gpu.ctx.buffers.create(
                Buffer::from_slice<u32>(gpu.allocator,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        zeros_lights, "light_flags")
                        .value());
        res.prefix = gpu.ctx.buffers.create(
                Buffer::from_slice<u32>(gpu.allocator,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        zeros_lights, "light_prefix")
                        .value());

        res.compact_lights = gpu.ctx.buffers.create(
                Buffer::zeroes(gpu.allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               sizeof(PointLight) * opts.light_count, "compact_lights")
                        .value());
    }

    // --- Clustering buffers ---
    {
        res.clustering_config = cluster_config(16, 9, 16, 0.1F, 1000.0F);

        constexpr u32 max_lights_per_cluster = 128u;
        res.max_light_indices = res.clustering_config.cluster_count * max_lights_per_cluster;

        std::vector<u32> zero_counts(res.clustering_config.cluster_count, 0u);
        res.cluster_counts = gpu.ctx.buffers.create(
                Buffer::from_slice<u32>(gpu.allocator,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        zero_counts, "cluster_counts")
                        .value());

        struct Cluster {
            u32 light_offset;
            u32 light_count;
        };

        struct LightVisibility {
            u32 x0, x1, y0, y1, z0, z1, is_visible, _pad;
        };

        res.visibility = gpu.ctx.buffers.create(Buffer::zeroes(gpu.allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                               sizeof(LightVisibility) * res.light_count,
                                                               "light_visibility_buffer")
                                                        .value());

        res.clusters =
                gpu.ctx.buffers.create(Buffer::zeroes(gpu.allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                      sizeof(Cluster) * res.clustering_config.cluster_count, "clusters")
                                               .value());

        std::vector<u32> zero_counters(res.clustering_config.cluster_count, 0u);
        res.cluster_counters = gpu.ctx.buffers.create(
                Buffer::from_slice<u32>(gpu.allocator,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        zero_counters, "cluster_counters")
                        .value());

        res.cluster_light_indices =
                gpu.ctx.buffers.create(Buffer::zeroes(gpu.allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                      sizeof(u32) * res.max_light_indices, "cluster_light_indices")
                                               .value());

        res.global_index_count = gpu.ctx.buffers.create(
                Buffer::from_value<u32>(gpu.allocator,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0u,
                                        "global_index_count")
                        .value());
    }

    // --- Frame UBO ring ---
    res.frame_ubo_ring = std::move(AlignedRingBuffer<FrameUBO>::create(gpu.ctx, "aligned_frame_ubo_buffer").value());

    // --- Indirect/draw streams ---
    res.indirect_ring = std::move(AlignedRingBuffer<VkDrawIndexedIndirectCommand>::create(
                                          gpu.ctx, AppResources::max_draws_per_frame,
                                          VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, "frame_indirect_cmds")
                                          .value());

    res.draw_material_id_ring =
            std::move(AlignedRingBuffer<u32>::create(gpu.ctx, AppResources::max_draws_per_frame,
                                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "frame_draw_material_ids")
                              .value());

    set_window_callbacks(gpu.window, ui);
    wire_event_dispatch(gpu.window, ui);

    glfwShowWindow(gpu.window);
    glfwFocusWindow(gpu.window);

    VkExtent2D last_extent = current_extent(gpu.window);
    ResizeGraph resize_graph{};
    u32 pipelines_node{};

    {
        const auto swapchain_node =
                resize_graph.add_node("swapchain", [&](VkExtent2D new_extent, const ResizeContext &) {
                    if (auto r = gpu.swapchain.recreate(new_extent); !r) {
                        vk_check(r.error());
                    }
                });

        const auto tonemapped_node =
                resize_graph.add_node("tonemapped_image", [&](VkExtent2D e, const ResizeContext &resize_context) {
                    const auto old_tonemap = res.tonemapped;

                    res.tonemapped = gpu.ctx.create_texture(create_offscreen_target(
                            gpu.allocator, e.width, e.height, VK_FORMAT_R8G8B8A8_UNORM, {}, "tonemapped"));
                    destroy(gpu.ctx, old_tonemap, resize_context.retire_value);
                });

        const auto offscreen_node = resize_graph.add_node("offscreen_targets", [&](VkExtent2D e,
                                                                                   const ResizeContext &rc) {
            const auto old_g0 = res.gbuffer0;
            const auto old_g1 = res.gbuffer1;
            const auto old_g2 = res.gbuffer2;
            const auto old_culling = res.debug_culling;
            const auto old_hdr = res.lit_hdr;
            const auto old_depth = res.depth;

            res.gbuffer0 = gpu.ctx.create_texture(create_offscreen_target(
                    gpu.allocator, e.width, e.height, VK_FORMAT_R8G8B8A8_UNORM, {}, "gbuffer0_albedo_ao"));

            res.gbuffer1 = gpu.ctx.create_texture(create_offscreen_target(gpu.allocator, e.width, e.height,
                                                                          VK_FORMAT_R16G16B16A16_SFLOAT, {},
                                                                          "gbuffer1_normal_rough_metal"));

            res.gbuffer2 = gpu.ctx.create_texture(create_offscreen_target(
                    gpu.allocator, e.width, e.height, VK_FORMAT_R16G16B16A16_SFLOAT, {}, "gbuffer2_emissive"));

            res.debug_culling = gpu.ctx.create_texture(create_offscreen_target(
                    gpu.allocator, e.width, e.height, VK_FORMAT_R16G16B16A16_SFLOAT, {}, "debug_culling"));

            res.depth = gpu.ctx.create_texture(create_depth_target(
                    gpu.allocator, e.width, e.height, VK_FORMAT_D32_SFLOAT, VK_SAMPLE_COUNT_1_BIT, false, "depth"));

            res.lit_hdr = gpu.ctx.create_texture(create_offscreen_target(gpu.allocator, e.width, e.height,
                                                                         VK_FORMAT_R16G16B16A16_SFLOAT, {}, "lit_hdr"));

            destroy(gpu.ctx, old_g0, rc.retire_value);
            destroy(gpu.ctx, old_g1, rc.retire_value);
            destroy(gpu.ctx, old_g2, rc.retire_value);
            destroy(gpu.ctx, old_hdr, rc.retire_value);
            destroy(gpu.ctx, old_depth, rc.retire_value);
            destroy(gpu.ctx, old_culling, rc.retire_value);
        });

        const auto uniforms_node = resize_graph.add_node("frame_ubo_camera", [&](VkExtent2D, const ResizeContext &) {
            // no-op for now
        });

        pipelines_node = resize_graph.add_node(
                "pipelines",
                [&](VkExtent2D, const ResizeContext &rc) {
                    const auto old_gbuffer_pipeline_lighting = pipes.gbuffer_pipeline_lighting;
                    const auto old_cube_rotation_pipeline = pipes.cube_rotation_pipeline;
                    const auto old_gbuffer_pipeline_mrt = pipes.gbuffer_pipeline_mrt;
                    const auto old_flags_pipeline = pipes.flags_pipeline;
                    const auto old_compact_pipeline = pipes.compact_pipeline;
                    const auto old_predepth_pipeline = pipes.predepth_pipeline;
                    const auto old_predepth_alpha_pipeline = pipes.predepth_alpha_pipeline;
                    const auto old_tonemap_pipeline = pipes.tonemap_pipeline;
                    const auto old_cluster_build_groups_pipeline = pipes.cluster_build_groups_pipeline;
                    const auto old_present_pipeline = pipes.present_pipeline;

                    std::array<const std::string_view, 2> names = {"LightFlagsCS", "LightCompactCS"};
                    std::array<ReflectionData, names.size()> reflection_data = {};
                    TRY_UNWRAP_WITH_DISCARD(culling_code,
                                            gpu.compiler->compile_from_file("shaders/light_cull_compact_modern.slang",
                                                                            std::span(names),
                                                                            std::span(reflection_data)),
                                            "Failed to compile light culling shader");

                    std::array<const std::string_view, 1> clustered_culling_names = {"BuildClusterCS"};
                    std::array<ReflectionData, clustered_culling_names.size()> clustered_culling_reflection_data = {};
                    TRY_UNWRAP_WITH_DISCARD(clustered_culling_code,
                                            gpu.compiler->compile_from_file(
                                                    "shaders/clustering.slang", std::span(clustered_culling_names),
                                                    std::span(clustered_culling_reflection_data)),
                                            "Failed to compile light clustering shader");

                    std::array<const std::string_view, 2> predepth_names{"main_vs_mdi", "fs_main"};
                    std::array<ReflectionData, predepth_names.size()> predepth_reflection{};
                    TRY_UNWRAP_WITH_DISCARD(predepth_code,
                                            gpu.compiler->compile_from_file("shaders/predepth.slang",
                                                                            std::span(predepth_names),
                                                                            std::span(predepth_reflection)),
                                            "Failed to compile predepth shader");

                    std::array<const std::string_view, 2> tonemap_names{"vs_main", "fs_main"};
                    std::array<ReflectionData, tonemap_names.size()> tonemap_reflection{};
                    TRY_UNWRAP_WITH_DISCARD(tonemap_code,
                                            gpu.compiler->compile_from_file("shaders/tonemap.slang",
                                                                            std::span(tonemap_names),
                                                                            std::span(tonemap_reflection)),
                                            "Failed to compile tonemap shader");

                    std::array<const std::string_view, 1> rotate_cubes_names{"rotate_cs"};
                    std::array<ReflectionData, rotate_cubes_names.size()> rotate_cubes_reflection{};
                    TRY_UNWRAP_WITH_DISCARD(rotate_cubes_code,
                                            gpu.compiler->compile_from_file("shaders/rotate_cubes.slang",
                                                                            std::span(rotate_cubes_names),
                                                                            std::span(rotate_cubes_reflection)),
                                            "Failed to compile rotate cubes shader");

                    std::array<const std::string_view, 4> gbuffer_entry_point_names = {
                            "main_vs_mdi", "main_fs_mdi", "vs_fullscreen_main", "fs_fullscreen_main"};
                    std::array<ReflectionData, gbuffer_entry_point_names.size()> gbuffer_reflection{};
                    TRY_UNWRAP_WITH_DISCARD(gbuffer_mrt_and_lighting_code,
                                            gpu.compiler->compile_from_file("shaders/gbuffer.slang",
                                                                            std::span(gbuffer_entry_point_names),
                                                                            std::span(gbuffer_reflection)),
                                            "Failed to compile gbuffer shader");

                    std::array<const std::string_view, 1> present_names = {"present_fs"};
                    std::array<ReflectionData, present_names.size()> present_reflection{};
                    TRY_UNWRAP_WITH_DISCARD(present_code,
                                            gpu.compiler->compile_from_file("shaders/present.slang",
                                                                            std::span(present_names),
                                                                            std::span(present_reflection)),
                                            "Failed to compile present shader");

                    auto &&[fp, cp] = create_compute_pipelines(gpu.device, *gpu.ctx.pipeline_cache, gpu.bindless.layout,
                                                               {}, std::span(culling_code), std::span(names));

                    auto &&[crp] = create_compute_pipelines(
                            gpu.device, *gpu.ctx.pipeline_cache, gpu.bindless.layout, sizeof(RotateCubesPushConstant),
                            std::span(rotate_cubes_code), std::span(rotate_cubes_names));

                    auto &&[cl_groups] = create_compute_pipelines(
                            gpu.device, *gpu.ctx.pipeline_cache, gpu.bindless.layout,
                            sizeof(ClusteredLightCullingPushConstants), std::span(clustered_culling_code),
                            std::span(clustered_culling_names));

                    auto gbuffer_pipeline = create_gbuffer_pipeline(
                            gpu.device, *gpu.ctx.pipeline_cache, gpu.bindless.layout,
                            gbuffer_mrt_and_lighting_code.at(0), gbuffer_mrt_and_lighting_code.at(1),
                            VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                            VK_FORMAT_D32_SFLOAT);

                    auto gbuf_light = create_deferred_lighting_graphics_pipeline(
                            gpu.device, *gpu.ctx.pipeline_cache, gpu.bindless.layout,
                            gbuffer_mrt_and_lighting_code.at(2), gbuffer_mrt_and_lighting_code.at(3),
                            "vs_fullscreen_main", "fs_fullscreen_main", VK_FORMAT_R16G16B16A16_SFLOAT);

                    auto pp = create_predepth_pipeline(gpu.device, *gpu.ctx.pipeline_cache, gpu.bindless.layout,
                                                       predepth_code.at(0), VK_FORMAT_D32_SFLOAT, gpu.msaa_samples);
                    auto pp_alpha = create_predepth_pipeline(gpu.device, *gpu.ctx.pipeline_cache, gpu.bindless.layout,
                                                             predepth_code.at(0), predepth_code.at(1),
                                                             VK_FORMAT_D32_SFLOAT, gpu.msaa_samples);

                    auto tp = create_fullscreen_pipeline(FullscreenPipelineInfo{
                            .device = gpu.device,
                            .cache = *gpu.ctx.pipeline_cache,
                            .bindless_layout = gpu.bindless.layout,
                            .fullscreen_vs = *gpu.ctx.shaders.get(pipes.fullscreen_vs),
                            .frag_code = tonemap_code.at(1),
                            .fs_entry = "fs_main",
                            .color_format = VK_FORMAT_R8G8B8A8_UNORM,
                            .push_constant_size = sizeof(TonemapPushConstants),
                            .enable_blend = false,
                    });

                    auto present_pipe = create_fullscreen_pipeline(FullscreenPipelineInfo{
                            .device = gpu.device,
                            .cache = *gpu.ctx.pipeline_cache,
                            .bindless_layout = gpu.bindless.layout,
                            .fullscreen_vs = *gpu.ctx.shaders.get(pipes.fullscreen_vs),
                            .frag_code = present_code.at(0),
                            .fs_entry = "present_fs",
                            .color_format = gpu.swapchain.format(),
                            .push_constant_size = sizeof(PresentPushConstants),
                            .enable_blend = false,
                    });

                    pipes.gbuffer_pipeline_lighting = gpu.ctx.create_pipeline(std::move(gbuf_light));
                    pipes.cube_rotation_pipeline = gpu.ctx.create_pipeline(std::move(crp));
                    pipes.gbuffer_pipeline_mrt = gpu.ctx.create_pipeline(std::move(gbuffer_pipeline));
                    pipes.flags_pipeline = gpu.ctx.create_pipeline(std::move(fp));
                    pipes.compact_pipeline = gpu.ctx.create_pipeline(std::move(cp));
                    pipes.predepth_pipeline = gpu.ctx.create_pipeline(std::move(pp));
                    pipes.predepth_alpha_pipeline = gpu.ctx.create_pipeline(std::move(pp_alpha));
                    pipes.tonemap_pipeline = gpu.ctx.create_pipeline(std::move(tp));
                    pipes.cluster_build_groups_pipeline = gpu.ctx.create_pipeline(std::move(cl_groups));
                    pipes.present_pipeline = gpu.ctx.create_pipeline(std::move(present_pipe));

                    destroy(gpu.ctx, old_gbuffer_pipeline_lighting, rc.retire_value);
                    destroy(gpu.ctx, old_cube_rotation_pipeline, rc.retire_value);
                    destroy(gpu.ctx, old_gbuffer_pipeline_mrt, rc.retire_value);
                    destroy(gpu.ctx, old_flags_pipeline, rc.retire_value);
                    destroy(gpu.ctx, old_compact_pipeline, rc.retire_value);
                    destroy(gpu.ctx, old_predepth_pipeline, rc.retire_value);
                    destroy(gpu.ctx, old_predepth_alpha_pipeline, rc.retire_value);
                    destroy(gpu.ctx, old_tonemap_pipeline, rc.retire_value);
                    destroy(gpu.ctx, old_cluster_build_groups_pipeline, rc.retire_value);
                    destroy(gpu.ctx, old_present_pipeline, rc.retire_value);
                },
                ResizeTrigger::Shaders);

        resize_graph.add_dependency(tonemapped_node, offscreen_node);
        resize_graph.add_dependency(offscreen_node, swapchain_node);
        resize_graph.add_dependency(pipelines_node, offscreen_node);
        resize_graph.add_dependency(uniforms_node, swapchain_node);
    }

    resize_graph.rebuild(last_extent, ResizeContext{.ctx = gpu.ctx, .retire_value = 0});

    // --- ImGui renderer + shader watcher ---
    ui.gui = std::make_unique<ImGuiRenderer>(gpu.window, static_cast<u32>(gpu.swapchain.image_count()), gpu.ctx,
                                             gpu.command_context, *gpu.compiler);

    auto gui_pipeline_node = resize_graph.add_node(
            "gui_pipeline", [&gui = *ui.gui](auto, const auto &) { gui.set_should_recompile(); },
            ResizeTrigger::Shaders);
    resize_graph.add_dependency(gui_pipeline_node, pipelines_node);

    ui.watcher = std::unique_ptr<efsw::FileWatcher, Deleter>(new efsw::FileWatcher(false), Deleter{});
    ui.listeners["update"] = std::unique_ptr<efsw::FileWatchListener, Deleter>(
            new ShaderSourceCodeChangeListener(&resize_graph), Deleter{});
    std::ignore = ui.watcher->addWatch("shaders", ui.listeners["update"].get(), true,
                                       {efsw::WatcherOption(efsw::Option::WinBufferSize, 128 * 1024)});
    ui.watcher->watch();

    // --- Graph setup ---
    if (!ui.graphs_initialized) {
        ui.gpu_frame_graph.add_line("Rotate");
        ui.gpu_frame_graph.add_line("Cull");
        ui.gpu_frame_graph.add_line("Clustering");
        ui.gpu_frame_graph.add_line("Pre-Depth");
        ui.gpu_frame_graph.add_line("GBuffer");
        ui.gpu_frame_graph.add_line("Deferred");
        ui.gpu_frame_graph.add_line("Tonemap");
        ui.gpu_frame_graph.add_line("Present");
        ui.graphs_initialized = true;
    }

    ui.last_frame_time = std::chrono::high_resolution_clock::now();

    auto stats = FrameStats{};

    while (!glfwWindowShouldClose(gpu.window)) {
        glfwPollEvents();

        const u64 completed_now = std::min(gpu.tl_compute.completed, gpu.tl_graphics.completed);
        glfwPollEvents();

        const auto extent = current_extent(gpu.window);
        const bool window_resized = (extent.width != last_extent.width || extent.height != last_extent.height);

        ResizeTrigger manual_trigger = resize_graph.get_and_clear_triggers();

        if (window_resized || manual_trigger != ResizeTrigger::None) {
            if (extent.width == 0 || extent.height == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            last_extent = extent;

            res.clustering_config = cluster_config(16, 9, 16, 0.1F, 1000.0F);

            ResizeTrigger final_trigger = manual_trigger;
            if (window_resized) {
                final_trigger = final_trigger | ResizeTrigger::Extent;
            }

            info("Resize trigger to rebuild: {}", to_string(final_trigger));

            resize_graph.rebuild(extent, ResizeContext{.ctx = gpu.ctx, .retire_value = completed_now}, final_trigger);

            if (window_resized) {
                continue;
            }
        }

        // timing
        auto current_frame_time = std::chrono::high_resolution_clock::now();
        ui.dt = std::chrono::duration<double>(current_frame_time - ui.last_frame_time).count();
        ui.last_frame_time = current_frame_time;

        const auto frame_extent = gpu.swapchain.extent();
        auto start_time = std::chrono::high_resolution_clock::now();

        const auto bounded_frame_index = static_cast<u32>(ui.frame_index % frames_in_flight);
        const auto last_frame_index = static_cast<u32>((ui.frame_index + frames_in_flight - 1u) % frames_in_flight);

        res.draw_stream.begin_frame();

        // camera update + frame ubo write
        ui.app_state.cam.update(gpu.window, ui.dt, ui.app_state.cam_in);
        constexpr float fov_y = glm::radians(70.0f);
        constexpr float z_near = 0.1f;

        write_camera_to_frame_ubo(gpu.ctx, res.frame_ubo_ring, bounded_frame_index, ui.app_state.cam, frame_extent,
                                  fov_y, z_near);

        ui.total_time += ui.dt;
        {
            constexpr auto rads_per_second = glm::radians(20.0f);
            const auto angle = static_cast<float>(ui.total_time * rads_per_second);

            const glm::vec3 sun_dir = glm::normalize(glm::vec3(std::cos(angle), std::sin(angle), -0.4f));
            auto sun_direction_intensity = glm::vec4(sun_dir, 1.5f);

            auto offset = offsetof(FrameUBO, sun_direction_intensity);
            res.frame_ubo_ring.write_field(gpu.ctx, bounded_frame_index, sun_direction_intensity, offset);
        }

        // mesh indirect
        const auto ranges = write_mesh_indirect(gpu.ctx, bounded_frame_index, res.draw_stream.writer, res.indirect_ring,
                                                res.draw_material_id_ring, res.mesh.mesh, res.instance_count, 0u);

        // bindless repopulate triggers shader rebuild
        if (gpu.bindless.repopulate_if_needed(gpu.ctx.textures, gpu.ctx.samplers)) {
            resize_graph.rebuild(current_extent(gpu.window),
                                 ResizeContext{.ctx = gpu.ctx, .retire_value = completed_now}, ResizeTrigger::Shaders);
            info("Bindless set was repopulated, resizing pipelines.");
        }

        auto &fs = res.frames[bounded_frame_index];

        ui.gui->begin_frame(ImGuiFramebuffer(extent, gpu.ctx.texture_format(res.tonemapped), gpu.swapchain.format(),
                                             gpu.swapchain.color_space()));

        draw_ui(ui.gpu_frame_graph, gpu.ctx, pipes.compute_query_pool, pipes.compute_stats_pool,
                pipes.graphics_query_pool, pipes.graphics_stats_pool, bounded_frame_index);

        if (fs.frame_done_value > 0) {
            VkSemaphoreWaitInfo wi{
                    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .semaphoreCount = 1,
                    .pSemaphores = &gpu.tl_graphics.timeline,
                    .pValues = &fs.frame_done_value,
            };
            vk_check(vkWaitSemaphores(gpu.device, &wi, UINT64_MAX));

            auto &&[a, b, c, d] = gpu.ctx.query_pools.get_multiple(
                    pipes.compute_query_pool[bounded_frame_index], pipes.graphics_query_pool[bounded_frame_index],
                    pipes.graphics_stats_pool[bounded_frame_index], pipes.compute_stats_pool[bounded_frame_index]);

            vkResetQueryPool(gpu.device, a->pool, 0, a->query_count);
            vkResetQueryPool(gpu.device, b->pool, 0, b->query_count);
            vkResetQueryPool(gpu.device, c->pool, 0, c->query_count);
            vkResetQueryPool(gpu.device, d->pool, 0, d->query_count);
        }

        // acquire
        auto acquired = gpu.swapchain.acquire_next_image(bounded_frame_index);
        if (!acquired) {
            const VkResult res_vk = acquired.error();
            if (res_vk == VK_ERROR_OUT_OF_DATE_KHR) {
                continue;
            }
            vk_check(res_vk);
        }

        const auto swap_image_index = acquired->image_index;
        const auto frame_sync = acquired->sync;

        // Precompute device addresses used in push constants
        const auto flags_addr = gpu.ctx.device_address(res.flags);
        const auto prefix_addr = gpu.ctx.device_address(res.prefix);
        const auto compact_addr = gpu.ctx.device_address(res.compact_lights);
        const auto culled_light_count_addr = gpu.ctx.device_address(res.culled_light_count);
        const auto point_lights_base_addr = gpu.ctx.device_address(res.point_lights_base);

        const auto cluster_counts_addr = gpu.ctx.device_address(res.cluster_counts);
        const auto clusters_addr = gpu.ctx.device_address(res.clusters);
        const auto cluster_counters_addr = gpu.ctx.device_address(res.cluster_counters);
        const auto cluster_light_indices_addr = gpu.ctx.device_address(res.cluster_light_indices);
        const auto global_index_count_addr = gpu.ctx.device_address(res.global_index_count);
        const auto visibility_addr = gpu.ctx.device_address(res.visibility);

        auto begin_query_for_index = [&c = gpu.ctx](const auto &cmd, GraphicsIndex index,
                                                    auto &stats_pool) -> VkQueryPool {
            u32 query_idx = static_cast<u32>(index);
            const auto *qs = c.query_pools.get(stats_pool);
            vkCmdBeginQuery(cmd, qs->pool, query_idx, 0);
            return qs->pool;
        };
        auto end_query_for_index = [](const auto &cmd, GraphicsIndex index, VkQueryPool pool) -> void {
            u32 query_idx = static_cast<u32>(index);
            vkCmdEndQuery(cmd, pool, query_idx);
        };

        auto rotate_cubes_gpu_val = submit_stage(
                gpu.tl_compute, gpu.device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(gpu.tracy_compute.ctx, cmd, "RotateCubesGPU");

                    auto &&[ts, stats_pool] =
                            gpu.ctx.query_pools.get_multiple(pipes.compute_query_pool[bounded_frame_index],
                                                             pipes.compute_stats_pool[bounded_frame_index]);

                    auto *pipe = gpu.ctx.pipeline_pool.get(pipes.cube_rotation_pipeline);

                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, ComputeStamp::RotateBegin);
                    begin_stats(cmd, *stats_pool, ComputeIndex::Rotate);

                    auto *cube_buffer = gpu.ctx.buffers.get(res.transforms_ring.handle());

                    RotateCubesPushConstant pc{
                            .cube_count = res.instance_count,
                            .delta_time = static_cast<float>(ui.dt),
                            .rads_per_second = glm::radians(20.0f),
                            .total_time = static_cast<f32>(ui.total_time),
                            .light_count = static_cast<u32>(res.all_point_lights.size()),
                            .transforms = res.transforms_ring.slot_device_address(bounded_frame_index),
                            .previous_frame_transforms = res.transforms_ring.slot_device_address(last_frame_index),
                            .point_lights = res.point_lights_ring.slot_device_address(bounded_frame_index),
                            .previous_point_lights = res.point_lights_ring.slot_device_address(last_frame_index),
                            .static_point_light_base = point_lights_base_addr,
                    };

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline);
                    vkCmdPushConstants(cmd, pipe->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

                    const u32 work_items = std::max(res.instance_count, pc.light_count);
                    const u32 groups = (work_items + 63u) / 64u;
                    vkCmdDispatch(cmd, groups, 1, 1);

                    end_stats(cmd, *stats_pool, ComputeIndex::Rotate);
                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, ComputeStamp::RotateEnd);

                    std::array<VkBufferMemoryBarrier2, 2> barriers{};
                    barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                    barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    barriers[0].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
                    barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    barriers[0].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                    barriers[0].buffer = cube_buffer->buffer();
                    barriers[0].offset =
                            static_cast<VkDeviceSize>(res.transforms_ring.slot_offset_bytes(bounded_frame_index));
                    barriers[0].size = static_cast<VkDeviceSize>(res.instance_count * sizeof(glm::mat4x3));

                    barriers[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                    barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    barriers[1].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
                    barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    barriers[1].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                    barriers[1].buffer = gpu.ctx.buffers.get(res.point_lights_ring.handle())->buffer();
                    barriers[1].offset = 0;
                    barriers[1].size = VK_WHOLE_SIZE;

                    VkDependencyInfo dep_info{};
                    dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dep_info.bufferMemoryBarrierCount = static_cast<u32>(barriers.size());
                    dep_info.pBufferMemoryBarriers = barriers.data();
                    vkCmdPipelineBarrier2(cmd, &dep_info);
                },
                no_waits);

        fs.timeline_values[stage_index(Stage::CubeRotation)] = rotate_cubes_gpu_val;

        const std::array cube_rotate_waits{TimelineWait{
                .value = fs.timeline_values[stage_index(Stage::CubeRotation)],
                .semaphore = gpu.tl_compute.timeline,
                .stage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
        }};

        auto predepth_val = submit_stage(
                gpu.tl_graphics, gpu.device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(gpu.tracy_graphics.ctx, cmd, "Predepth");

                    auto &&[ts, stats_pool] =
                            gpu.ctx.query_pools.get_multiple(pipes.graphics_query_pool[bounded_frame_index],
                                                             pipes.graphics_stats_pool[bounded_frame_index]);

                    auto &&[predepth, alpha] =
                            gpu.ctx.pipeline_pool.get_multiple(pipes.predepth_pipeline, pipes.predepth_alpha_pipeline);

                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, GraphicsStamp::PreDepthBegin);
                    begin_stats(cmd, *stats_pool, GraphicsIndex::PreDepth);

                    auto &&depth = gpu.ctx.textures.get(res.depth);
                    auto &&[indirect, verts, idx, materials] =
                            gpu.ctx.buffers.get_multiple(res.indirect_ring.handle(), res.mesh.pos_uv_buffer,
                                                         res.mesh.index_buffer, res.mesh.material_buffer);

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

                    const PredepthPushConstants pc{
                            .ubo = res.frame_ubo_ring.slot_device_address(bounded_frame_index),
                            .transforms = res.transforms_ring.slot_device_address(bounded_frame_index),
                            .draw_material_ids = res.draw_material_id_ring.slot_device_address(bounded_frame_index),
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
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gpu.bindless.pipeline_layout, 0, 1,
                                            &gpu.bindless.set, 0, nullptr);

                    if (ranges.opaque_count > 0) {
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, predepth->pipeline);

                        PredepthPushConstants opaque_pc = pc;
                        opaque_pc.base_draw_id = ranges.opaque_base;

                        vkCmdPushConstants(cmd, predepth->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(opaque_pc),
                                           &opaque_pc);

                        const VkDeviceSize opaque_offset =
                                static_cast<VkDeviceSize>(res.indirect_ring.slot_offset_bytes(bounded_frame_index)) +
                                (ranges.opaque_base * sizeof(VkDrawIndexedIndirectCommand));

                        vkCmdDrawIndexedIndirect(cmd, indirect->buffer(), opaque_offset, ranges.opaque_count,
                                                 sizeof(VkDrawIndexedIndirectCommand));
                    }

                    if (ranges.alpha_count > 0) {
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, alpha->pipeline);

                        PredepthPushConstants alpha_pc = pc;
                        alpha_pc.base_draw_id = ranges.alpha_base;

                        vkCmdPushConstants(cmd, alpha->layout,
                                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                           sizeof(alpha_pc), &alpha_pc);

                        const VkDeviceSize alpha_offset =
                                static_cast<VkDeviceSize>(res.indirect_ring.slot_offset_bytes(bounded_frame_index)) +
                                (ranges.alpha_base * sizeof(VkDrawIndexedIndirectCommand));

                        vkCmdDrawIndexedIndirect(cmd, indirect->buffer(), alpha_offset, ranges.alpha_count,
                                                 sizeof(VkDrawIndexedIndirectCommand));
                    }

                    vkCmdEndRendering(cmd);
                    end_stats(cmd, *stats_pool, GraphicsIndex::PreDepth);
                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, GraphicsStamp::PreDepthEnd);
                },
                SubmitSynchronisation{.timeline_waits = cube_rotate_waits});

        fs.timeline_values[stage_index(Stage::Predepth)] = predepth_val;


        const std::array culling_waits{
                TimelineWait{.value = fs.timeline_values[stage_index(Stage::Predepth)],
                             .semaphore = gpu.tl_graphics.timeline,
                             .stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
                TimelineWait{.value = fs.timeline_values[stage_index(Stage::CubeRotation)],
                             .semaphore = gpu.tl_compute.timeline,
                             .stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
        };
        auto light_val = submit_stage(
                gpu.tl_compute, gpu.device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(gpu.tracy_compute.ctx, cmd, "LightCulling");

                    auto &&[cqs, css] = gpu.ctx.query_pools.get_multiple(pipes.compute_query_pool[bounded_frame_index],
                                                                         pipes.compute_stats_pool[bounded_frame_index]);

                    write_ts(cmd, *cqs, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, ComputeStamp::CullBegin);
                    begin_stats(cmd, *css, ComputeIndex::Cull);

                    const PointLightCullingPushConstants pc{
                            .ubo = res.frame_ubo_ring.slot_device_address(bounded_frame_index),
                            .lights = res.point_lights_ring.slot_device_address(bounded_frame_index),
                            .flags = flags_addr,
                            .prefix = prefix_addr,
                            .compact = compact_addr,
                            .culled_light_count = culled_light_count_addr,
                            .light_count = res.light_count,
                    };

                    auto bind_and_dispatch = [&](auto &pl, u32 groups_x) {
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.layout, 0, 1, &gpu.bindless.set,
                                                0, nullptr);

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

                    fill_zeros(cmd, gpu.ctx.buffers, res.flags, res.prefix, res.compact_lights, res.culled_light_count);

                    vkCmdPipelineBarrier2(cmd, &dep_info);

                    const u32 gc = (res.light_count + THREADS_PER_GROUP - 1) / THREADS_PER_GROUP;

                    auto &&[flags, compact] =
                            gpu.ctx.pipeline_pool.get_multiple(pipes.flags_pipeline, pipes.compact_pipeline);

                    bind_and_dispatch(*flags, gc);
                    vkCmdPipelineBarrier2(cmd, &dep_info);

                    bind_and_dispatch(*compact, gc);

                    end_stats(cmd, *css, ComputeIndex::Cull);
                    write_ts(cmd, *cqs, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, ComputeStamp::CullEnd);
                },
                SubmitSynchronisation{.timeline_waits = culling_waits});
        fs.timeline_values[stage_index(Stage::LightCulling)] = light_val;

        const std::array clustering_waits{TimelineWait{.value = fs.timeline_values[stage_index(Stage::LightCulling)],
                                                       .semaphore = gpu.tl_compute.timeline,
                                                       .stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT}};
        auto light_clustering_val = submit_stage(
                gpu.tl_compute, gpu.device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(gpu.tracy_compute.ctx, cmd, "ClusteredLightCulling");

                    auto &&[cqs, css] = gpu.ctx.query_pools.get_multiple(pipes.compute_query_pool[bounded_frame_index],
                                                                         pipes.compute_stats_pool[bounded_frame_index]);

                    write_ts(cmd, *cqs, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, ComputeStamp::ClusteringBegin);
                    begin_stats(cmd, *css, ComputeIndex::Clustering);

                    const ClusteredLightCullingPushConstants pc{
                            .frame_ubo = res.frame_ubo_ring.slot_device_address(bounded_frame_index),
                            .culled_lights = compact_addr,
                            .culled_light_count = culled_light_count_addr,

                            .z_near = res.clustering_config.z_near,
                            .z_far = res.clustering_config.z_far,
                            .log_z_scale = res.clustering_config.log_z_scale,

                            .tiles_x = res.clustering_config.tiles_x,
                            .tiles_y = res.clustering_config.tiles_y,
                            .tiles_z = res.clustering_config.tiles_z,
                            .cluster_count = res.clustering_config.cluster_count,

                            .visibility = visibility_addr,
                            .cluster_counts = cluster_counts_addr,
                            .clusters = clusters_addr,
                            .cluster_counters = cluster_counters_addr,
                            .cluster_light_indices = cluster_light_indices_addr,
                            .global_index_count = global_index_count_addr,
                    };

                    auto build_pipe = gpu.ctx.pipeline_pool.get(pipes.cluster_build_groups_pipeline);

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, build_pipe->pipeline);
                    vkCmdPushConstants(cmd, build_pipe->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                    vkCmdDispatch(cmd, res.clustering_config.cluster_count, 1, 1);

                    end_stats(cmd, *css, ComputeIndex::Clustering);
                    write_ts(cmd, *cqs, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, ComputeStamp::ClusteringEnd);
                    TracyVkCollect(gpu.tracy_compute.ctx, cmd);
                },
                SubmitSynchronisation{.timeline_waits = clustering_waits});

        fs.timeline_values[stage_index(Stage::LightClustering)] = light_clustering_val;

        const std::array gbuffer_waits{
                TimelineWait{
                        .value = fs.timeline_values[stage_index(Stage::CubeRotation)],
                        .semaphore = gpu.tl_compute.timeline,
                        .stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                },
                TimelineWait{
                        .value = fs.timeline_values[stage_index(Stage::Predepth)],
                        .semaphore = gpu.tl_graphics.timeline,
                        .stage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, // depth test
                },
        };


        auto gbuffer_val = submit_stage(
                gpu.tl_graphics, gpu.device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(gpu.tracy_graphics.ctx, cmd, "GBuffer MRT");

                    auto *ts = gpu.ctx.query_pools.get(pipes.graphics_query_pool[bounded_frame_index]);
                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, GraphicsStamp::GbufferBegin);
                    auto *pool = begin_query_for_index(cmd, GraphicsIndex::GBuffer,
                                                       pipes.graphics_stats_pool[bounded_frame_index]);

                    auto *mrt_pipeline = gpu.ctx.pipeline_pool.get(pipes.gbuffer_pipeline_mrt);

                    auto *g0 = gpu.ctx.textures.get(res.gbuffer0);
                    auto *g1 = gpu.ctx.textures.get(res.gbuffer1);
                    auto *g2 = gpu.ctx.textures.get(res.gbuffer2);
                    auto *depth = gpu.ctx.textures.get(res.depth);

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
                            gpu.ctx.buffers.get_multiple(res.indirect_ring.handle(), res.mesh.vertex_buffer,
                                                         res.mesh.index_buffer, res.mesh.material_buffer);

                    RenderingPushConstants pc{
                            .ubo = res.frame_ubo_ring.slot_device_address(bounded_frame_index),
                            .transforms = res.transforms_ring.slot_device_address(bounded_frame_index),
                            .draw_material_ids = res.draw_material_id_ring.slot_device_address(bounded_frame_index),
                            .materials = materials->device_address(),
                            .base_draw_id = ranges.opaque_base,
                            .sampler_index = pipes.linear_repeat.index(),
                    };

                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mrt_pipeline->layout, 0, 1,
                                            &gpu.bindless.set, 0, nullptr);

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
                                static_cast<VkDeviceSize>(res.indirect_ring.slot_offset_bytes(bounded_frame_index)) +
                                static_cast<VkDeviceSize>(ranges.opaque_base) * sizeof(VkDrawIndexedIndirectCommand);

                        vkCmdDrawIndexedIndirect(cmd, indirect->buffer(), indirect_offset_bytes, ranges.opaque_count,
                                                 sizeof(VkDrawIndexedIndirectCommand));
                    }

                    if (ranges.alpha_count > 0) {
                        pc.base_draw_id = ranges.alpha_base;
                        vkCmdPushConstants(cmd, mrt_pipeline->layout,
                                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc),
                                           &pc);

                        VkDeviceSize indirect_offset_bytes =
                                static_cast<VkDeviceSize>(res.indirect_ring.slot_offset_bytes(bounded_frame_index)) +
                                static_cast<VkDeviceSize>(ranges.alpha_base) * sizeof(VkDrawIndexedIndirectCommand);

                        vkCmdDrawIndexedIndirect(cmd, indirect->buffer(), indirect_offset_bytes, ranges.alpha_count,
                                                 sizeof(VkDrawIndexedIndirectCommand));
                    }

                    vkCmdEndRendering(cmd);

                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, GraphicsStamp::GbufferEnd);
                    end_query_for_index(cmd, GraphicsIndex::GBuffer, pool);
                },
                SubmitSynchronisation{.timeline_waits = gbuffer_waits});
        fs.timeline_values[stage_index(Stage::GBuffer)] = gbuffer_val;

        const std::array deferred_waits{
                TimelineWait{
                        .value = fs.timeline_values[stage_index(Stage::GBuffer)],
                        .semaphore = gpu.tl_graphics.timeline,
                        .stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                },
                TimelineWait{
                        .value = fs.timeline_values[stage_index(Stage::LightClustering)],
                        .semaphore = gpu.tl_compute.timeline,
                        .stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                },
        };

        auto deferred_val = submit_stage(
                gpu.tl_graphics, gpu.device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(gpu.tracy_graphics.ctx, cmd, "DeferredLighting(FS)");

                    auto &&ts = gpu.ctx.query_pools.get(pipes.graphics_query_pool[bounded_frame_index]);
                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, GraphicsStamp::DeferredBegin);
                    auto *pool = begin_query_for_index(cmd, GraphicsIndex::Deferred,
                                                       pipes.graphics_stats_pool[bounded_frame_index]);

                    auto mrt_lighting = gpu.ctx.pipeline_pool.get(pipes.gbuffer_pipeline_lighting);

                    auto *g0 = gpu.ctx.textures.get(res.gbuffer0);
                    auto *g1 = gpu.ctx.textures.get(res.gbuffer1);
                    auto *g2 = gpu.ctx.textures.get(res.gbuffer2);
                    auto *depth = gpu.ctx.textures.get(res.depth);
                    auto *lit = gpu.ctx.textures.get(res.lit_hdr);

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
                                            &gpu.bindless.set, 0, nullptr);

                    auto &&[vp, sc] = viewport_scissors(frame_extent);
                    vkCmdSetViewport(cmd, 0, 1, &vp);
                    vkCmdSetScissor(cmd, 0, 1, &sc);

                    DeferredLightingPushConstants pc{
                            .frame_ubo = res.frame_ubo_ring.slot_device_address(bounded_frame_index),
                            .point_lights = res.point_lights_ring.slot_device_address(bounded_frame_index),

                            .tiles_x = res.clustering_config.tiles_x,
                            .tiles_y = res.clustering_config.tiles_y,
                            .tiles_z = res.clustering_config.tiles_z,
                            .log_z_scale = res.clustering_config.log_z_scale,

                            .clusters = clusters_addr,
                            .cluster_light_indices = cluster_light_indices_addr,

                            .gbuffer0_index = res.gbuffer0.index(),
                            .gbuffer1_index = res.gbuffer1.index(),
                            .gbuffer2_index = res.gbuffer2.index(),
                            .depth_index = res.depth.index(),
                            .lit_hdr_uav_index = 0,
                            .debug_output_index = res.debug_culling.index(),
                            .sampler_index = pipes.linear_clamp.index(),
                            .debug_mode = static_cast<u32>(ui.debug_mode),
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
                    end_query_for_index(cmd, GraphicsIndex::Deferred, pool);
                },
                SubmitSynchronisation{.timeline_waits = deferred_waits});

        fs.timeline_values[stage_index(Stage::DeferredLighting)] = deferred_val;

        const std::array tonemap_waits{
                TimelineWait{
                        .value = fs.timeline_values[stage_index(Stage::DeferredLighting)],
                        .semaphore = gpu.tl_graphics.timeline,
                        .stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                },
        };

        auto tonemap_val = submit_stage(
                gpu.tl_graphics, gpu.device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(gpu.tracy_graphics.ctx, cmd, "Tonemapping");

                    auto &&ts = gpu.ctx.query_pools.get(pipes.graphics_query_pool[bounded_frame_index]);
                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, GraphicsStamp::TonemapBegin);
                    auto *pool = begin_query_for_index(cmd, GraphicsIndex::Tonemap,
                                                       pipes.graphics_stats_pool[bounded_frame_index]);


                    auto *tonemap = gpu.ctx.pipeline_pool.get(pipes.tonemap_pipeline);

                    auto &&hdr = gpu.ctx.textures.get(res.lit_hdr);
                    auto &&ldr = gpu.ctx.textures.get(res.tonemapped);

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

                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tonemap->layout, 0, 1,
                                            &gpu.bindless.set, 0, nullptr);

                    float exposure = 1.0f;
                    TonemapPushConstants pc{
                            .exposure = exposure,
                            .image_index = res.lit_hdr.index(),
                            .sampler_index = pipes.linear_clamp.index(),
                    };


                    auto &&[vp, sc] = viewport_scissors(frame_extent);
                    vkCmdSetViewport(cmd, 0, 1, &vp);
                    vkCmdSetScissor(cmd, 0, 1, &sc);

                    vkCmdPushConstants(cmd, tonemap->layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                       0, sizeof(pc), &pc);

                    vkCmdDraw(cmd, 3, 1, 0, 0);

                    vkCmdEndRendering(cmd);

                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, GraphicsStamp::TonemapEnd);
                    end_query_for_index(cmd, GraphicsIndex::Tonemap, pool);
                },
                SubmitSynchronisation{.timeline_waits = tonemap_waits});

        fs.timeline_values[stage_index(Stage::Tonemapping)] = tonemap_val;


        const std::array imgui_waits{
                TimelineWait{
                        .value = fs.timeline_values[stage_index(Stage::Tonemapping)],
                        .semaphore = gpu.tl_graphics.timeline,
                },
        };
        auto imgui_val = submit_stage(
                gpu.tl_graphics, gpu.device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(gpu.tracy_graphics.ctx, cmd, "ImGui");

                    auto &&ts = gpu.ctx.query_pools.get(pipes.graphics_query_pool[bounded_frame_index]);
                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, GraphicsStamp::UIBegin);
                    auto *pool = begin_query_for_index(cmd, GraphicsIndex::UI,
                                                       pipes.graphics_stats_pool[bounded_frame_index]);

                    auto &&ldr = gpu.ctx.textures.get(res.tonemapped);

                    // Transition tonemapped image for ImGui rendering
                    ldr->transition(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

                    VkRenderingAttachmentInfo color_attachment{};
                    color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    color_attachment.imageView = ldr->sampled_view;
                    color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

                    VkRenderingInfo ri{};
                    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                    ri.renderArea.offset = {.x = 0, .y = 0};
                    ri.renderArea.extent = {.width = frame_extent.width, .height = frame_extent.height};
                    ri.layerCount = 1;
                    ri.colorAttachmentCount = 1;
                    ri.pColorAttachments = &color_attachment;

                    vkCmdBeginRendering(cmd, &ri);

                    ui.gui->end_frame(cmd);

                    vkCmdEndRendering(cmd);

                    ldr->transition(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_IMAGE_LAYOUT_GENERAL, // or PRESENT_SRC_KHR if final
                                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                                    VK_ACCESS_2_NONE);

                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, GraphicsStamp::UIEnd);
                    end_query_for_index(cmd, GraphicsIndex::UI, pool);
                },
                SubmitSynchronisation{.timeline_waits = imgui_waits});

        fs.timeline_values[stage_index(Stage::UI)] = imgui_val;


        const std::array present_timeline_waits = {
                TimelineWait{
                        .value = fs.timeline_values[stage_index(Stage::UI)],
                        .semaphore = gpu.tl_graphics.timeline,
                },
        };

        const std::array present_binary_waits{BinaryWait{
                .semaphore = frame_sync.image_available,
                .stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        }};

        const std::array present_binary_signals{frame_sync.render_finished};

        auto swapchain_val = submit_stage(
                gpu.tl_graphics, gpu.device,
                [&](VkCommandBuffer cmd) {
                    TRACY_GPU_ZONE(gpu.tracy_graphics.ctx, cmd, "PresentFullscreen");

                    auto *pool = begin_query_for_index(cmd, GraphicsIndex::Present,
                                                       pipes.graphics_stats_pool[bounded_frame_index]);

                    auto &&tonemapped = gpu.ctx.textures.get(res.tonemapped);
                    auto *present = gpu.ctx.pipeline_pool.get(pipes.present_pipeline);

                    VkImage dst_image = gpu.swapchain.image(swap_image_index);
                    VkImageView dst_view = gpu.swapchain.image_view(swap_image_index); // MUST be identity swizzle

                    auto ts = gpu.ctx.query_pools.get(pipes.graphics_query_pool[bounded_frame_index]);
                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, GraphicsStamp::PresentBegin);

                    // Transition tonemapped to shader-read
                    tonemapped->transition(
                            cmd, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

                    // Transition swapchain to color attachment
                    auto to_color = create_info<VkImageMemoryBarrier2>();
                    to_color.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                    to_color.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
                    to_color.srcAccessMask = VK_ACCESS_2_NONE;
                    to_color.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                    to_color.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                    to_color.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    to_color.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    to_color.image = dst_image;
                    to_color.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

                    auto dep_to_color = create_info<VkDependencyInfo>();
                    dep_to_color.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dep_to_color.imageMemoryBarrierCount = 1;
                    dep_to_color.pImageMemoryBarriers = &to_color;
                    vkCmdPipelineBarrier2(cmd, &dep_to_color);

                    auto color_attachment = create_info<VkRenderingAttachmentInfo>();
                    color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    color_attachment.imageView = dst_view;
                    color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

                    auto ri = create_info<VkRenderingInfo>();
                    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                    ri.renderArea = {.offset = {0, 0}, .extent = gpu.swapchain.extent()};
                    ri.layerCount = 1;
                    ri.colorAttachmentCount = 1;
                    ri.pColorAttachments = &color_attachment;

                    vkCmdBeginRendering(cmd, &ri);

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, present->pipeline);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, present->layout, 0, 1,
                                            &gpu.bindless.set, 0, nullptr);

                    auto &&[vp, sc] = viewport_scissors(gpu.swapchain.extent());
                    vkCmdSetViewport(cmd, 0, 1, &vp);
                    vkCmdSetScissor(cmd, 0, 1, &sc);
                    const bool swap_is_srgb = gpu.swapchain.format() == VK_FORMAT_B8G8R8A8_SRGB ||
                                              gpu.swapchain.format() == VK_FORMAT_R8G8B8A8_SRGB;

                    PresentPushConstants pc{
                            .image_index = res.tonemapped.index(),
                            .sampler_index = pipes.linear_clamp.index(),
                            .dst_is_srgb = swap_is_srgb ? 1u : 0u,
                    };

                    vkCmdPushConstants(cmd, present->layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                       0, sizeof(pc), &pc);

                    vkCmdDraw(cmd, 3, 1, 0, 0);


                    vkCmdEndRendering(cmd);

                    VkImageMemoryBarrier2 to_present{
                            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                            .pNext = nullptr,
                            .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                            .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                            .dstStageMask = VK_PIPELINE_STAGE_2_NONE,
                            .dstAccessMask = VK_ACCESS_2_NONE,
                            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .image = dst_image,
                            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                    };
                    auto dep_to_present = create_info<VkDependencyInfo>();
                    dep_to_present.imageMemoryBarrierCount = 1;
                    dep_to_present.pImageMemoryBarriers = &to_present;
                    vkCmdPipelineBarrier2(cmd, &dep_to_present);

                    write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, GraphicsStamp::PresentEnd);
                    end_query_for_index(cmd, GraphicsIndex::Present, pool);

                    TracyVkCollect(gpu.tracy_graphics.ctx, cmd);
                },
                SubmitSynchronisation{
                        .timeline_waits = present_timeline_waits,
                        .binary_waits = present_binary_waits,
                        .binary_signals = present_binary_signals,
                });

        fs.frame_done_value = swapchain_val;
        const auto completed = std::min(gpu.tl_compute.completed, gpu.tl_graphics.completed);
        gpu.ctx.destroy_queue.retire(completed);

        auto frame_end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(frame_end - start_time).count();
        stats.add_sample(ms);

        const VkResult present_res =
                gpu.swapchain.present(gpu.graphics_queue, swap_image_index, frame_sync.render_finished);
        FrameMark;
        if (present_res == VK_ERROR_OUT_OF_DATE_KHR || present_res == VK_SUBOPTIMAL_KHR) {
            auto result = gpu.swapchain.recreate(current_extent(gpu.window));
            if (!result)
                vk_check(result.error());
        } else {
            vk_check(present_res);
        }

        ui.frame_index++;
    }

    info("Light count {}", opts.light_count);
    info("frames: {}", stats.samples.size());
    info("mean/frametime:   {:.3f} ms", stats.mean);
    info("median: {:.3f} ms", stats.median());
    info("stddev: {:.3f} ms", stats.stddev_sample());
    info("quartiles: {}", stats.quartiles());
    info("Total: {:.3f} s", stats.total() / 1000.0F);

    vkDeviceWaitIdle(gpu.device);

    ui.gui.reset();
    gpu.ctx.clear_all();

    gpu.compiler.reset();

    ui.watcher.reset();
    ui.listeners.clear();

    gpu.ctx.destroy_queue.retire(UINT64_MAX);

    gpu.tracy_compute.shutdown();
    gpu.tracy_graphics.shutdown();

    destruction::global_command_context(gpu.command_context);
    destruction::bindless_set(gpu.bindless);
    destruction::timelines(gpu.device, gpu.tl_graphics, gpu.tl_transfer, gpu.tl_compute);
    destruction::allocator(gpu.allocator);
    destruction::swapchain(gpu.swapchain);
    destruction::wsi(instance.instance, gpu.surface, gpu.window);
    destruction::device(gpu.device);
    destruction::instance(instance);
    volkFinalize();
    glfwTerminate();

    return 0;
}
