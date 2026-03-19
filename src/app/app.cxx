#include "app/app.hxx"

#include <3PP/stb_image.h>
#include <GLFW/glfw3.h>
#include <chrono>
#include <csignal>
#include <deque>
#include <efsw/efsw.hpp>
#include <execution>
#include <future>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/packing.hpp>
#include <imgui.h>
#include <iostream>
#include <random>
#include <ranges>
#include <thread>
#include <tl/expected.hpp>
#include <unordered_map>

#include "Constants.hxx"
#include "DeviceThreadPool.hxx"
#include "ImGuizmo.h"
#include "Logger.hxx"
#include "Pipelines.hxx"
#include "RenderDoc.hxx"
#include "RenderSubmission.hxx"
#include "SceneLoader.hxx"
#include "Types.hxx"
#include "app/icon_parser.hxx"
#include "app/render.hxx"
#include "app/render_passes.hxx"
#include "app/ui.hxx"
#include "scene/Components.hxx"
#include "scene/Scene.hxx"

extern auto ImGui_KeyToImGuiKey(int key) -> ImGuiKey;

static volatile sig_atomic_t keep_running = 1;
static void sig_handler(int) { keep_running = 0; }


namespace {
    constexpr auto poll_streamer = [](AppResources &res, AppGpuState &gpu) {
        if (res.icons_loaded) {
            return;
        }

        auto result = res.asset_streamer->poll(gpu.graphics_queue);
        if (!result) {
            error("Asset streaming failed: {}", result.error().message);
            res.icons_loaded = true;
            return;
        }

        if (*result) {
            res.icons_loaded = true;
            info("All assets streamed");
        }
    };

    [[nodiscard]] auto stbi_channels_for(IconLoadDescription::Channels channels) -> int {
        switch (channels) {
            case IconLoadDescription::Channels::r:
                return STBI_grey;
            case IconLoadDescription::Channels::rg:
                return STBI_grey_alpha;
            case IconLoadDescription::Channels::rgb:
                return STBI_rgb;
            case IconLoadDescription::Channels::rgba:
                return STBI_rgb_alpha;
        }
        return STBI_rgb_alpha;
    }

    /*auto generate_sky_cubemap(RenderContext &ctx, VmaAllocator alloc,
                          VkDevice device, VkDescriptorSetLayout bindless_layout,
                          VkDescriptorSet bindless_set,
                          GlobalCommandContext &cmd_ctx,
                          Compiler &compiler,
                          glm::vec3 sun_direction, float sun_intensity,
                          u32 face_size) -> tl::expected<TextureHandle, Error> {

    struct AtmospherePushConstants {
        glm::vec3 sun_direction;
        float     sun_intensity;
        u32       output_cubemap_face;
        u32       output_image_index;
        u32       face_size;
    };

    // Compile shader
    constexpr std::array names = {std::string_view{"generate_sky_cs"}};
    std::array<ReflectionData, names.size()> reflection_data{};
    TRY_PROPAGATE(sky_code,
                            compiler.compile_from_file("assets/shaders/atmosphere_equirect.slang",
                                                       std::span(names),
                                                       std::span(reflection_data)),
                            "Failed to compile sky generation shader");

    // Create transient pipeline
    auto [pipeline, layout] = create_compute_pipeline(
            device, nullptr, bindless_layout,
            sky_code.at(0),
            sizeof(AtmospherePushConstants),
            "generate_sky_cs");

    // Create the cubemap storage target
    OffscreenTarget sky_target = create_offscreen_target(
            alloc, face_size, face_size,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_SAMPLE_COUNT_1_BIT,
            TargetSamplerConfiguration{
                .sampled_storage_transfer = {0b111},
                .dims = {
                    .mip_levels   = 1,
                    .array_layers = 6,
                    .view_type    = VK_IMAGE_VIEW_TYPE_CUBE,
                },
            },
            "sky_cubemap");

        auto sky_handle         = ctx.textures.create(std::move(sky_target));
    auto *sky_tex           = ctx.textures.get(sky_handle);
    const u32 storage_index = sky_handle.index();

    for (u32 face = 0; face < 6; ++face) {
        submit_one_time_cmd(cmd_ctx, [&](VkCommandBuffer cmd) {
            AtmospherePushConstants pc{
                .sun_direction       = glm::normalize(sun_direction),
                .sun_intensity       = sun_intensity,
                .output_cubemap_face = face,
                .output_image_index  = storage_index,
                .face_size           = face_size,
            };

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    layout, 0, 1, &bindless_set, 0, nullptr);
            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(pc), &pc);

            const u32 groups = (face_size + 7) / 8;
            vkCmdDispatch(cmd, groups, groups, 1);
        }, true);
    }

    submit_one_time_cmd(cmd_ctx, [&](VkCommandBuffer cmd) {
        auto barrier = create_info<VkImageMemoryBarrier2>();
        barrier.srcStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask       = VK_ACCESS_2_SHADER_WRITE_BIT;
        barrier.dstStageMask        = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        barrier.dstAccessMask       = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = sky_tex->image;
        barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};

        auto dep = create_info<VkDependencyInfo>();
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }, true);

    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, layout, nullptr);

    OffscreenTarget& result = *sky_tex;
    result.initialized     = true;
    return sky_handle;
}*/
} // namespace


namespace {
    GLFWkeyfun imgui_key_callback = nullptr;
    GLFWcharfun imgui_char_callback = nullptr;
    GLFWmousebuttonfun imgui_mouse_button_callback = nullptr;
    GLFWcursorposfun imgui_cursor_pos_callback = nullptr;
    GLFWscrollfun imgui_scroll_callback = nullptr;

    auto update_mouse_delta(AppState &app, glm::vec2 pos) -> glm::vec2 {
        glm::vec2 delta{0.0f};

        if (!app.mouse_inited) {
            app.last_mouse = pos;
            app.mouse_inited = true;
            return delta;
        }

        delta = pos - app.last_mouse;
        app.last_mouse = pos;
        return delta;
    }

    auto vi(const AppState &app) -> const auto & { return app.viewport_input; }

    auto route_keyboard_to_app(AppState const &app) -> bool {
        return vi(app).focused && !vi(app).imgui_blocks_keyboard;
    }

    auto route_text_to_app(AppState const &app) -> bool { return vi(app).focused && !vi(app).imgui_blocks_keyboard; }

    auto route_mouse_to_app(AppState const &app) -> bool { return vi(app).hovered && !vi(app).imgui_blocks_mouse; }

    auto set_window_callbacks(GLFWwindow *window, AppUI &ui) -> void {
        glfwSetWindowUserPointer(window, &ui.app_state);

        // Detach any existing callbacks (ImGui backend may have installed them),
        // store them so we can forward into them.
        imgui_key_callback = glfwSetKeyCallback(window, nullptr);
        imgui_char_callback = glfwSetCharCallback(window, nullptr);
        imgui_mouse_button_callback = glfwSetMouseButtonCallback(window, nullptr);
        imgui_cursor_pos_callback = glfwSetCursorPosCallback(window, nullptr);
        imgui_scroll_callback = glfwSetScrollCallback(window, nullptr);

        glfwSetKeyCallback(window, [](GLFWwindow *w, int key, int scancode, int action, int mods) {
            if (imgui_key_callback) {
                imgui_key_callback(w, key, scancode, action, mods);
            }

            auto &app = *static_cast<AppState *>(glfwGetWindowUserPointer(w));

            if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
                glfwSetWindowShouldClose(w, GLFW_TRUE);
                return;
            }

            if (!route_keyboard_to_app(app)) {
                return;
            }

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
        });

        glfwSetCharCallback(window, [](GLFWwindow *w, unsigned int c) {
            if (imgui_char_callback) {
                imgui_char_callback(w, c);
            }

            auto &app = *static_cast<AppState *>(glfwGetWindowUserPointer(w));

            if (!route_text_to_app(app)) {
                return;
            }

            auto event = std::make_unique<CharInputEvent>();
            event->codepoint = c;
            app.event_system.push_event(std::move(event));
        });

        glfwSetMouseButtonCallback(window, [](GLFWwindow *w, int button, int action, int mods) {
            if (imgui_mouse_button_callback)
                imgui_mouse_button_callback(w, button, action, mods);

            auto &app = *static_cast<AppState *>(glfwGetWindowUserPointer(w));

            const bool alt_down =
                    glfwGetKey(w, GLFW_KEY_LEFT_ALT) == GLFW_PRESS || glfwGetKey(w, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;
            const bool super_down = glfwGetKey(w, GLFW_KEY_LEFT_SUPER) == GLFW_PRESS ||
                                    glfwGetKey(w, GLFW_KEY_RIGHT_SUPER) == GLFW_PRESS;
            const bool modifier_down = alt_down || super_down;
            const bool gizmo_wants_mouse = ImGuizmo::IsOver() || ImGuizmo::IsUsing();

            if (button == GLFW_MOUSE_BUTTON_RIGHT) {
                if (action == GLFW_PRESS && app.viewport_input.hovered && !gizmo_wants_mouse) {
                    app.cam_in.rmb = true;
                    begin_cursor_capture(w, app);
                } else if (action == GLFW_RELEASE && app.cam_in.rmb) {
                    app.cam_in.rmb = false;
                    end_cursor_capture(w, app);
                }
                return;
            }

            if (button == GLFW_MOUSE_BUTTON_LEFT && modifier_down) {
                if (action == GLFW_PRESS && app.viewport_input.hovered && !gizmo_wants_mouse) {
                    app.cam_in.lmb = true;
                    app.cam_in.orbit_capture = true;
                    begin_cursor_capture(w, app);
                } else if (action == GLFW_RELEASE && app.cam_in.orbit_capture) {
                    app.cam_in.lmb = false;
                    app.cam_in.orbit_capture = false;
                    end_cursor_capture(w, app);
                }
                return;
            }

            const bool camera_capturing = app.cam_in.rmb || app.cam_in.orbit_capture;
            if (!camera_capturing && !route_mouse_to_app(app))
                return;

            if (action == GLFW_PRESS) {
                auto e = std::make_unique<MouseButtonPressedEvent>();
                e->button = button;
                e->mods = mods;
                app.event_system.push_event(std::move(e));
            } else if (action == GLFW_RELEASE) {
                auto e = std::make_unique<MouseButtonReleasedEvent>();
                e->button = button;
                e->mods = mods;
                app.event_system.push_event(std::move(e));
            }
        });

        glfwSetCursorPosCallback(window, [](GLFWwindow *w, double x, double y) {
            if (imgui_cursor_pos_callback) {
                imgui_cursor_pos_callback(w, x, y);
            }

            auto &app = *static_cast<AppState *>(glfwGetWindowUserPointer(w));
            const glm::vec2 pos{static_cast<float>(x), static_cast<float>(y)};

            // In DISABLED mode, pos is already a virtual 'unbounded' coordinate
            const glm::vec2 delta = update_mouse_delta(app, pos);

            const bool capturing = app.cam_in.rmb || app.cam_in.orbit_capture;

            if (capturing) {
                if (delta.x != 0.0f || delta.y != 0.0f) {
                    auto e = std::make_unique<CursorMovedEvent>();
                    e->position = pos;
                    e->delta = delta;
                    app.event_system.push_event(std::move(e));
                }
                // REMOVED: warp_to_center logic. GLFW_CURSOR_DISABLED does this for you.
                return;
            }

            if (!route_mouse_to_app(app))
                return;

            auto e = std::make_unique<CursorMovedEvent>();
            e->position = pos;
            e->delta = delta;
            app.event_system.push_event(std::move(e));
        });


        glfwSetScrollCallback(window, [](GLFWwindow *w, double xoff, double yoff) {
            if (imgui_scroll_callback) {
                imgui_scroll_callback(w, xoff, yoff);
            }

            auto &app = *static_cast<AppState *>(glfwGetWindowUserPointer(w));
            const bool capturing = app.cam_in.rmb || app.cam_in.orbit_capture;

            if (!capturing && !route_mouse_to_app(app)) {
                return;
            }

            auto e = std::make_unique<ScrollEvent>();
            e->x_offset = static_cast<float>(xoff);
            e->y_offset = static_cast<float>(yoff);
            app.event_system.push_event(std::move(e));
        });

        glfwSetWindowSizeCallback(window, [](GLFWwindow *w, int, int) {
            auto &data = *static_cast<AppState *>(glfwGetWindowUserPointer(w));
            data.resized = true;
        });

        glfwSetFramebufferSizeCallback(window, [](GLFWwindow *w, int, int) {
            auto &data = *static_cast<AppState *>(glfwGetWindowUserPointer(w));
            data.resized = true;
        });
    }

    auto poll_gamepad(AppState &app) -> void {
        GLFWgamepadstate pad{};
        if (!glfwGetGamepadState(GLFW_JOYSTICK_1, &pad)) {
            return;
        }

        // B button: toggle orbit/fly (edge detect to avoid spamming)
        if (pad.buttons[GLFW_GAMEPAD_BUTTON_B] == GLFW_PRESS && !app.cam_in.gamepad_b_prev) {
            auto e = std::make_unique<GamepadButtonPressedEvent>();
            e->button = GLFW_GAMEPAD_BUTTON_B;
            app.event_system.push_event(std::move(e));
        }
        app.cam_in.gamepad_b_prev = pad.buttons[GLFW_GAMEPAD_BUTTON_B] == GLFW_PRESS;

        // Axes go straight into cam_in — no events needed, same as mouse_delta
        auto deadzone = [](float v, float dz) -> float {
            if (std::abs(v) < dz)
                return 0.0f;
            return (v - std::copysign(dz, v)) / (1.0f - dz);
        };

        constexpr float dz = 0.15f;
        app.cam_in.gamepad_left = {deadzone(pad.axes[GLFW_GAMEPAD_AXIS_LEFT_X], dz),
                                   deadzone(pad.axes[GLFW_GAMEPAD_AXIS_LEFT_Y], dz)};
        app.cam_in.gamepad_right = {deadzone(pad.axes[GLFW_GAMEPAD_AXIS_RIGHT_X], dz),
                                    deadzone(pad.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y], dz)};
        app.cam_in.gamepad_rt = (pad.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER] + 1.0f) * 0.5f;
        app.cam_in.gamepad_lb = pad.buttons[GLFW_GAMEPAD_BUTTON_LEFT_BUMPER] == GLFW_PRESS;
        app.cam_in.gamepad_rb = pad.buttons[GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER] == GLFW_PRESS;
    }
} // namespace


namespace {
    auto wire_event_dispatch(AppUI &ui) -> void {
        ui.app_state.event_system.set_event_callback([&](Event &e) {
            EventDispatcher dispatcher(e);

            dispatcher.dispatch<KeyPressedEvent>([&](KeyPressedEvent &event) {
                if (event.key == GLFW_KEY_F1) {
                    ui.debug_mode = AppUI::ClusterDebugMode::ClusterGrid;
                    return true;
                } else if (event.key == GLFW_KEY_F2) {
                    ui.debug_mode = AppUI::ClusterDebugMode::LightCount;
                    return true;
                } else if (event.key == GLFW_KEY_F3) {
                    ui.debug_mode = AppUI::ClusterDebugMode::LightDensity;
                    return true;
                } else if (event.key == GLFW_KEY_F4) {
                    ui.debug_mode = AppUI::ClusterDebugMode::DepthSlices;
                    return true;
                } else if (event.key == GLFW_KEY_F5) {
                    ui.debug_mode = AppUI::ClusterDebugMode::LightHeatmap;
                    return true;
                } else if (event.key == GLFW_KEY_F6) {
                    ui.debug_mode = AppUI::ClusterDebugMode::FirstLight;
                    return true;
                } else if (event.key == GLFW_KEY_F7) {
                    ui.debug_mode = AppUI::ClusterDebugMode::ClusterOccupancy;
                    return true;
                } else if (event.key == GLFW_KEY_F8) {
                    ui.debug_mode = AppUI::ClusterDebugMode::None;
                    return true;
                }
                if (event.key == GLFW_KEY_F11) {
                    ui.capture_next_frame = true;
                    return true;
                }

                return false;
            });

            dispatcher.dispatch<MouseButtonPressedEvent>([&](MouseButtonPressedEvent &event) {
                if (ImGuizmo::IsOver()) {
                    return true; // Event handled by gizmo
                }
                if (event.button == GLFW_MOUSE_BUTTON_LEFT) {
                    ui.app_state.cam_in.lmb = true;
                    return true;
                } else if (event.button == GLFW_MOUSE_BUTTON_MIDDLE) {
                    ui.app_state.cam_in.mmb = true;
                    return true;
                } else if (event.button == GLFW_MOUSE_BUTTON_RIGHT) {
                    ui.app_state.cam_in.rmb = true;
                    return true;
                }
                return false;
            });

            dispatcher.dispatch<MouseButtonReleasedEvent>([&](MouseButtonReleasedEvent &event) {
                if (event.button == GLFW_MOUSE_BUTTON_LEFT) {
                    ui.app_state.cam_in.lmb = false;
                    return true;
                } else if (event.button == GLFW_MOUSE_BUTTON_MIDDLE) {
                    ui.app_state.cam_in.mmb = false;
                    return true;
                } else if (event.button == GLFW_MOUSE_BUTTON_RIGHT) {
                    ui.app_state.cam_in.rmb = false;
                    return true;
                }
                return false;
            });

            dispatcher.dispatch<CursorMovedEvent>([&](CursorMovedEvent &event) {
                ui.app_state.cam_in.mouse_delta += event.delta;
                return true;
            });

            dispatcher.dispatch<ScrollEvent>([&](ScrollEvent &event) {
                ui.app_state.cam_in.scroll_delta += event.y_offset;
                return true;
            });
        });
    }
} // namespace

namespace {

    auto generate_random_hierarchies(entt::registry &reg, std::span<const entt::entity> entities, int k,
                                     std::mt19937 &rng, float leaf_probability = 0.3f) -> std::vector<entt::entity> {
        std::uniform_real_distribution<float> chance(0.0f, 1.0f);

        std::vector<std::vector<entt::entity>> depth_buckets(k + 1);
        depth_buckets[0].push_back(entities[0]);

        for (size_t i = 1; i < entities.size(); ++i) {
            int target_depth;
            if (chance(rng) < leaf_probability) {
                target_depth = k;
            } else {
                std::uniform_int_distribution<int> depth_dist(1, k);
                target_depth = depth_dist(rng);
            }

            while (target_depth > 0 && depth_buckets[target_depth - 1].empty())
                --target_depth;

            if (target_depth == 0) {
                depth_buckets[0].push_back(entities[i]);
                continue;
            }

            auto &parent_bucket = depth_buckets[target_depth - 1];
            std::uniform_int_distribution<int> parent_dist(0, (int) parent_bucket.size() - 1);
            auto parent = parent_bucket[parent_dist(rng)];

            auto &parent_hc = reg.get_or_emplace<HierarchyComponent>(parent);
            auto &child_hc = reg.get_or_emplace<HierarchyComponent>(entities[i]);
            parent_hc.children.push_back(entities[i]);
            child_hc.parent = parent;

            depth_buckets[target_depth].push_back(entities[i]);
        }

        return depth_buckets[0];
    }

    auto create_scene(Scene &scene) -> void {
        auto &reg = scene.registry;

        auto rng =
                std::mt19937{static_cast<unsigned long>(std::chrono::system_clock::now().time_since_epoch().count())};
        auto distrib = std::uniform_real_distribution<float>{-5.0f, 5.0f};

        auto sponza = reg.create();
        reg.emplace<MeshComponent>(sponza, MeshComponent{.name = "sponza", .mesh_index = 0u});
        reg.emplace<TransformComponent>(sponza, glm::identity<glm::mat4x3>());

        auto capsule = reg.create();
        reg.emplace<MeshComponent>(capsule, MeshComponent{.name = "capsule", .mesh_index = 1u});
        reg.emplace<TransformComponent>(capsule, glm::identity<glm::mat4x3>());

        std::vector<entt::entity> helmets;
        helmets.reserve(100);

        for (auto i: std::views::iota(0, 100)) {
            auto e = reg.create();
            reg.emplace<MeshComponent>(e, MeshComponent{.name = std::format("damaged_helmet_{}", i), .mesh_index = 2u});
            auto random_position = glm::vec3{distrib(rng), distrib(rng), distrib(rng)};
            auto tx = glm::translate(glm::mat4{1.0f}, random_position);
            reg.emplace<TransformComponent>(e, glm::mat4x3{std::move(tx)});
            helmets.push_back(e);
        }

        generate_random_hierarchies(reg, helmets, 4, rng, 0.3f);
    }

    template<typename T>
    auto flush_render_queue(WatermarkedQueue<T> &queue, AppResources &res, RenderContext &ctx, u32 frame_idx) -> void {
        res.mesh_instance_ranges.clear();

        std::ranges::stable_sort(queue.objects, [](const MeshSubmission &a, const MeshSubmission &b) {
            return a.mesh_index < b.mesh_index;
        });

        static thread_local std::vector<InstanceData> instance_scratch;
        instance_scratch.clear();

        u32 ring_offset = 0;

        auto flush_batch = [&](u32 mesh_index) {
            const u32 count = static_cast<u32>(instance_scratch.size());
            res.instance_ring.write_elements(ctx, frame_idx, ring_offset,
                                             std::span<const InstanceData>{instance_scratch});
            res.mesh_instance_ranges.push_back({
                    .mesh_index = mesh_index,
                    .instance_count = count,
                    .base_instance = ring_offset,
            });
            ring_offset += count;
            instance_scratch.clear();
        };

        for (u32 i = 0; i < queue.objects.size(); ++i) {
            const auto &sub = queue.objects[i];
            const bool new_batch = !instance_scratch.empty() && queue.objects[i - 1].mesh_index != sub.mesh_index;

            if (new_batch) {
                flush_batch(queue.objects[i - 1].mesh_index);
            }
            instance_scratch.push_back(InstanceData{sub.transform, sub.lod_level});
        }

        if (!instance_scratch.empty()) {
            flush_batch(queue.objects.back().mesh_index);
        }

        res.flushed_instance_count = static_cast<u32>(queue.objects.size());
        queue.reset();
    }
} // namespace

auto BindlessApp::run(CLIOptions &opts, InstanceWithDebug &instance, RenderDocContext *renderdoc)
        -> tl::expected<int, Error> {
    signal(SIGINT, sig_handler);

    AppGpuState gpu{};
    AppPipelines pipes{};
    AppResources res{};
    AppUI ui{};
    AppScene scene{};

    AppContext app_context{
            .gpu = gpu,
            .pipes = pipes,
            .res = res,
            .ui = ui,
            .scene = scene,
    };

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
        gpu.queue_family_indices = {
                .graphics = gfx_i,
                .compute = comp_i,
                .transfer = xfer_i,
        };

        auto &&[device, gfx_q, comp_q, xfer_q, enabled] =
                create_device(gpu.physical_device, gpu.queue_family_indices.graphics, gpu.queue_family_indices.compute,
                              gpu.queue_family_indices.transfer);

        gpu.device = device;
        gpu.graphics_queue = gfx_q;
        gpu.compute_queue = comp_q;
        gpu.transfer_queue = xfer_q;
        gpu.enabled_features = std::move(enabled);

        gpu.tracy_graphics.init_calibrated(instance, gpu.physical_device, gpu.device, "Graphics Queue");
        gpu.tracy_compute.init_calibrated(instance, gpu.physical_device, gpu.device, "Compute Queue");
    }


    {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

        gpu.window = glfwCreateWindow(static_cast<i32>(opts.width), static_cast<i32>(opts.height), "Bindless", nullptr,
                                      nullptr);
        if (!gpu.window) {
            error("Could not create window");
            return 1;
        }

        auto glfw_res = glfwCreateWindowSurface(instance.instance, gpu.window, nullptr, &gpu.surface);

        if (glfw_res != VK_SUCCESS) {
            const char *description;
            int code = glfwGetError(&description);

            if (description) {
                error("GLFW Error ({}): {}", code, description);
                std::abort();
            }
        }

        std::array<GLFWimage, 2> icons{};
        std::array<i32, 2> sizes{16, 48};
        i32 channels{4};
        icons.at(0) = {sizes.at(0), sizes.at(0),
                       stbi_load("assets/editor/icons/icon_small.png", &sizes.at(0), &sizes.at(0), &channels, 4)};
        icons.at(1) = {sizes.at(1), sizes.at(1),
                       stbi_load("assets/editor/icons/icon_large.png", &sizes.at(1), &sizes.at(1), &channels, 4)};

        if (icons.at(0).pixels != nullptr && icons.at(1).pixels != nullptr) {
            glfwSetWindowIcon(gpu.window, static_cast<i32>(icons.size()), icons.data());
            stbi_image_free(icons.at(0).pixels);
            stbi_image_free(icons.at(1).pixels);
        } else {
            error("Failed to load one or more window icons.");
        }

        std::ifstream f("assets/editor/gamecontrollerdb.txt");
        if (f) {
            std::stringstream buf;
            buf << f.rdbuf();
            ensure(glfwUpdateGamepadMappings(buf.str().c_str()), "Could not initialise gamepad inputs.");
            info("Initialised gamepad mappings.");
        }
    }


    {
        auto maybe_swapchain = Swapchain::create(SwapchainCreateInfo{
                .physical_device = gpu.physical_device,
                .device = gpu.device,
                .surface = gpu.surface,
                .graphics_family = gpu.queue_family_indices.graphics,
                .extent = VkExtent2D{opts.width, opts.height},
                .vsync = opts.vsync,
        });
        if (!maybe_swapchain) {
            return 1;
        }
        gpu.swapchain = std::move(maybe_swapchain.value());
    }

    // --- Command context + allocator + timelines ---
    gpu.allocator = create_allocator(instance.instance, gpu.physical_device, gpu.device, &gpu.enabled_features);

    gpu.tl_compute = create_compute_timeline(gpu.device, gpu.compute_queue, gpu.queue_family_indices.compute);
    gpu.tl_graphics = create_graphics_timeline(gpu.device, gpu.graphics_queue, gpu.queue_family_indices.graphics);
    gpu.tl_transfer = create_transfer_timeline(gpu.device, gpu.transfer_queue, gpu.queue_family_indices.transfer);

    gpu.bindless.init(gpu.device, query_bindless_caps(gpu.physical_device), 8u, 8u, 8u, 8u, 0u);
    gpu.bindless.grow_if_needed(300u, 40u, 32u, 8u);

    {
        res.asset_streamer = std::make_unique<AssetStreamer>(AssetStreamer::Config{
                .device = gpu.device,
                .queue_family = gpu.queue_family_indices.graphics,
                .submissions_per_frame = 2,
                .chunk_size = 4,
        });
    }

    {
        const VkSampleCountFlagBits requested = msaa_from_cli(opts.msaa);
        gpu.msaa_samples = clamp_msaa_samples(gpu.physical_device, requested);
        info("MSAA requested: {}, Engine supplied: {}", static_cast<u32>(requested),
             static_cast<u32>(gpu.msaa_samples));

        gpu.ctx = RenderContext{
                .allocator = gpu.allocator,
                .bindless_set = &gpu.bindless,
                .command_ctx =
                        create_global_cmd_context(gpu.device, gpu.graphics_queue, gpu.queue_family_indices.graphics),
                .pipeline_cache = std::make_unique<PipelineCache>(gpu.device, opts.pipeline_cache_dir),
                .queues =
                        {
                                .graphics = {.queue = gpu.graphics_queue,
                                             .family_index = gpu.queue_family_indices.graphics},
                                .compute = {.queue = gpu.compute_queue,
                                            .family_index = gpu.queue_family_indices.compute},
                                .transfer = {.queue = gpu.transfer_queue,
                                             .family_index = gpu.queue_family_indices.transfer},
                        },
        };

        pipes.fullscreen_vs = gpu.ctx.shaders.get_handle(Pipeline::get_or_create_fullscreen_vs(gpu.ctx));
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

            vkResetQueryPool(gpu.device, qpc, 0, compute_query_count);
            vkResetQueryPool(gpu.device, qpg, 0, graphics_query_count);
            vkResetQueryPool(gpu.device, stats_pool, 0, stats_graphics_count);
            vkResetQueryPool(gpu.device, compute_stats, 0, stats_compute_count);
        }
    }

    // --- Default textures (white/black/flat-normal) ---
    {
        std::array<u8, 4> white{255, 255, 255, 255};
        std::array<u8, 4> black{0, 0, 0, 255};
        std::array<u8, 4> flat_normal{128, 128, 255, 255};

        auto white_handle = gpu.ctx.create_texture(
                create_image_from_span_v2(gpu.allocator, *gpu.ctx.command_ctx, 1, 1, VK_FORMAT_R8G8B8A8_UNORM,
                                          std::as_bytes(std::span(white)), "white-texture"));
        auto black_handle = gpu.ctx.create_texture(
                create_image_from_span_v2(gpu.allocator, *gpu.ctx.command_ctx, 1, 1, VK_FORMAT_R8G8B8A8_UNORM,
                                          std::as_bytes(std::span(black)), "black-texture"));
        auto flat_normal_handle = gpu.ctx.create_texture(
                create_image_from_span_v2(gpu.allocator, *gpu.ctx.command_ctx, 1, 1, VK_FORMAT_R8G8B8A8_UNORM,
                                          std::as_bytes(std::span(flat_normal)), "flat-normals-texture"));

        // FIXME: This (these indices) is a requirement, a system wide precondition for every part.
#ifndef NDEBUG
        ASSERT(white_handle.index() == white_texture_index,
               "White texture was not assigned the expected bindless index");
        ASSERT(black_handle.index() == black_texture_index,
               "Black texture was not assigned the expected bindless index");
        ASSERT(flat_normal_handle.index() == normal_texture_index,
               "Flat normal texture was not assigned the expected bindless index");
#else
        (void) white_handle;
        (void) black_handle;
        (void) flat_normal_handle;
#endif
    }

    {
        auto _ = NanoProfiler{"Submitting icons"};
        for (const auto &icon: std::filesystem::directory_iterator("assets/editor/icons")) {
            if (icon.path().extension() != ".png")
                continue;

            const auto stem = icon.path().stem().string();
            auto parse_result = IconFilenameParser{stem}.parse();
            if (!parse_result) {
                continue;
            }

            const auto name = parse_result->name;
            const auto desc = parse_result->desc;
            const auto path = icon.path();
            auto staged = std::make_shared<std::optional<StagedImage>>();

            res.asset_streamer->submit(
                    [=, &gpu](VkCommandBuffer cmd) mutable -> tl::expected<void, Error> {
                        i32 w{}, h{}, c{};
                        auto *pixels = stbi_load(path.string().c_str(), &w, &h, &c, stbi_channels_for(desc.channels));
                        if (!pixels)
                            return tl::unexpected{Error::make_error(Error::Type::RenderError,
                                                                    "Failed to load icon '{}'", path.string())};

                        const auto pixel_count = static_cast<usize>(w * h * desc.bytes_per_pixel());
                        TRY_PROPAGATE(result,
                                      stage_image(gpu.allocator, cmd, static_cast<u32>(w), static_cast<u32>(h),
                                                  desc.vk_format(), std::span{pixels, pixel_count},
                                                  std::format("icon_{}", name)),
                                      "Could not stage icon");

                        stbi_image_free(pixels);
                        *staged = std::move(result);
                        return {};
                    },
                    [=, &res, &gpu]() {
                        res.icons_map[name] = gpu.ctx.create_texture(std::move(staged->value().target));
                    });
        }
        info("Queued {} icons for streaming", res.asset_streamer->pending_count());
    }

    {
        auto _ = NanoProfiler{"Perlin noise"};
        auto noise = generate_perlin(2048, 2048);
        res.perlin_noise =
                gpu.ctx.create_texture(create_image_from_span_v2(gpu.allocator, *gpu.ctx.command_ctx, 2048u, 2048u,
                                                                 VK_FORMAT_R8_UNORM, std::span{noise}, "perlin_noise"));
    }
    {
        auto _ = NanoProfiler{"SSAO Kernels"};
        auto ssao_hemisphere_kernel = []() -> std::array<glm::vec4, 32> {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            std::default_random_engine rng{std::random_device{}()};

            std::array<glm::vec4, 32> kernel{};
            for (u32 i = 0; i < 32; ++i) {
                glm::vec3 sample{
                        dist(rng) * 2.0f - 1.0f,
                        dist(rng) * 2.0f - 1.0f,
                        dist(rng),
                };
                sample = glm::normalize(sample);
                sample *= dist(rng);

                float scale = static_cast<float>(i) / 32.0f;
                scale = glm::mix(0.1f, 1.0f, scale * scale);
                sample *= scale;

                kernel[i] = glm::vec4(sample, 0.0f);
            }
            return kernel;
        }();

        auto ssao_noise_kernel = []() -> std::array<glm::vec4, 16> {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            std::default_random_engine rng{std::random_device{}()};

            std::array<glm::vec4, 16> noise{};
            for (auto &n: noise) {
                const float angle = dist(rng) * 2.0f * glm::pi<float>();
                n = glm::vec4(std::cos(angle), std::sin(angle), 0.0f, 0.0f);
            }
            return noise;
        }();

        res.ssao_hemisphere_kernel =
                gpu.ctx.create_buffer(Buffer::from_slice<glm::vec4>(gpu.allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                                    ssao_hemisphere_kernel, "ssao_hemisphere_kernel")
                                              .value());

        res.noise_ssao_kernel =
                gpu.ctx.create_buffer(Buffer::from_slice<glm::vec4>(gpu.allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                                    ssao_noise_kernel, "noise_ssao_kernel")
                                              .value());
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
            auto ci = create_info<VkSamplerCreateInfo>();
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

            gpu.ctx.create_sampler(ci, "noise_sampler");
        }

        {
            auto sampler_info = create_info<VkSamplerCreateInfo>();
            sampler_info.magFilter = VK_FILTER_LINEAR;
            sampler_info.minFilter = VK_FILTER_LINEAR;
            sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
            sampler_info.compareEnable = VK_TRUE;
            sampler_info.compareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
            sampler_info.minLod = 0.0f;
            sampler_info.maxLod = 0.0f;
            sampler_info.anisotropyEnable = VK_FALSE;
            sampler_info.unnormalizedCoordinates = VK_FALSE;

            pipes.depth_compare_filter = gpu.ctx.create_comparison_sampler(sampler_info, "depth_compare_filter");
        }
    }

    gpu.bindless.repopulate_if_needed(gpu.ctx.textures, gpu.ctx.samplers, gpu.ctx.comparison_samplers);

    {
        TRY_PROPAGATE(loaded_mesh, load_scene(gpu.ctx, "assets/meshes/SponzaGLTF/sponza_converted.scene.bz2", 0.01f),
                      "Failed to load cube mesh");
        res.meshes.emplace_back(std::move(loaded_mesh));

        TRY_PROPAGATE(loaded_capsule, load_static_mesh(gpu.ctx, "assets/meshes/capsule.obj"),
                      "Failed to load capsule mesh");
        res.meshes.emplace_back(std::move(loaded_capsule));

        TRY_PROPAGATE(loaded_damaged_helmet,
                      load_scene(gpu.ctx, "assets/meshes/DamagedHelmetGLTF/damaged_helmet_converted.scene.bz2"),
                      "Failed to load damaged helmet mesh");
        res.meshes.emplace_back(std::move(loaded_damaged_helmet));

        create_scene(scene.scene);
    }

    const auto graphics_family = gpu.queue_family_indices.graphics;
    const auto compute_family = gpu.queue_family_indices.compute;
    std::array<const u32, 2> family_indices = {graphics_family, compute_family};

    {
        res.all_point_lights = std::vector<PointLight>(opts.light_count);
        res.all_point_lights_zero = std::vector<PointLight>(opts.light_count);
        res.light_count = static_cast<u32>(res.all_point_lights.size());

        const auto mesh_aabb = res.meshes.at(0).mesh_aabb;
        spawn_lights_in_aabb(mesh_aabb, res.all_point_lights);


        res.point_lights_base = gpu.ctx.buffers.create(
                Buffer::from_slice<PointLight>(gpu.allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, res.all_point_lights,
                                               "base_static_point_lights", family_indices)
                        .value());

        auto ring = AlignedRingBuffer<PointLight>::create(gpu.ctx, res.light_count, VkBufferUsageFlags{},
                                                          "point_lights_ring", family_indices);
        res.point_lights_ring = std::move(ring.value());
        res.point_lights_ring.write_all_slots(gpu.ctx, res.all_point_lights);

        res.instance_ring = AlignedRingBuffer<InstanceData>::create(gpu.ctx, AppResources::max_draws_per_frame, {},
                                                                    "instances", family_indices)
                                    .value();
        res.instance_ring.write_all_slots(gpu.ctx, InstanceData::empty());

        res.prefix = AlignedRingBuffer<u32>::create(gpu.ctx, res.light_count, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                    "light_prefix", family_indices)
                             .value();

        res.culled_light_count = AlignedRingBuffer<u32>::create(gpu.ctx, 1, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                                "culled_light_count_ring", family_indices)
                                         .value();

        res.flags = AlignedRingBuffer<u32>::create(gpu.ctx, res.light_count, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                   "flags_ring", family_indices)
                            .value();

        res.compact_lights =
                AlignedRingBuffer<PointLight>::create(gpu.ctx, res.light_count, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                      "compact_lights_ring", family_indices)
                        .value();
    }

    {
        res.clustering_config = cluster_config(16, 9, 24, z_near, z_far);

        res.max_light_indices = res.clustering_config.cluster_count * max_lights_per_cluster;

        res.clusters = AlignedRingBuffer<Cluster>::create(gpu.ctx, res.clustering_config.cluster_count, 0,
                                                          "clusters_ring", family_indices)
                               .value();

        res.cluster_light_indices = AlignedRingBuffer<u32>::create(gpu.ctx, res.max_light_indices, 0,
                                                                   "cluster_light_indices_ring", family_indices)
                                            .value();

        ui.clustering_config = res.clustering_config;
    }

    res.frame_ubo_ring =
            std::move(AlignedRingBuffer<FrameUBO>::create(gpu.ctx, "aligned_frame_ubo_buffer", family_indices).value());
    res.indirect_ring = std::move(AlignedRingBuffer<VkDrawIndexedIndirectCommand>::create(
                                          gpu.ctx, AppResources::max_draws_per_frame,
                                          VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, "frame_indirect_cmds", family_indices)
                                          .value());
    res.mesh_indirect_ring =
            std::move(AlignedRingBuffer<VkDrawMeshTasksIndirectCommandEXT>::create(
                              gpu.ctx, AppResources::max_draws_per_frame, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                              "frame_indirect_mesh_cmds", family_indices)
                              .value());

    res.draw_material_id_ring = std::move(AlignedRingBuffer<u32>::create(gpu.ctx, AppResources::max_draws_per_frame,
                                                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                                         "frame_draw_material_ids", family_indices)
                                                  .value());

    set_window_callbacks(gpu.window, ui);
    wire_event_dispatch(ui);

    u32 pipelines_node{};
    {
        gpu.window_resize_graph.add_node("swapchain", [&](VkExtent2D new_extent, const ResizeContext &) {
            if (auto r = gpu.swapchain.recreate(new_extent); !r) {
                vk_check(r.error());
            }
        });

        pipelines_node = gpu.window_resize_graph.add_node(
                "pipelines",
                [&](VkExtent2D, const ResizeContext &rc) {
                    std::array<const std::string_view, 2> clustered_culling_names = {"BuildClusterCS",
                                                                                     "LightFinaliseCS"};
                    std::array<ReflectionData, clustered_culling_names.size()> clustered_culling_reflection_data = {};
                    TRY_UNWRAP_WITH_DISCARD(
                            clustered_culling_code,
                            gpu.compiler->compile_from_file("assets/shaders/clustering.slang",
                                                            std::span(clustered_culling_names),
                                                            std::span(clustered_culling_reflection_data)),
                            "Failed to compile light clustering shader");

                    std::array<const std::string_view, 2> predepth_names{"main_vs_mdi", "fs_main"};
                    std::array<ReflectionData, predepth_names.size()> predepth_reflection{};
                    TRY_UNWRAP_WITH_DISCARD(predepth_code,
                                            gpu.compiler->compile_from_file("assets/shaders/predepth.slang",
                                                                            std::span(predepth_names),
                                                                            std::span(predepth_reflection)),
                                            "Failed to compile predepth shader");

                    std::array<const std::string_view, 2> directional_shadow_map_names{"shadow_vs_mdi",
                                                                                       "shadow_fs_main"};
                    std::array<ReflectionData, directional_shadow_map_names.size()> directional_shadow_map_reflection{};
                    TRY_UNWRAP_WITH_DISCARD(
                            directional_shadow_map_code,
                            gpu.compiler->compile_from_file("assets/shaders/directional_shadow_map.slang",
                                                            std::span(directional_shadow_map_names),
                                                            std::span(directional_shadow_map_reflection)),
                            "Failed to compile directional shadow map shader");

                    std::array<const std::string_view, 2> tonemap_names{"vs_main", "fs_main"};
                    std::array<ReflectionData, tonemap_names.size()> tonemap_reflection{};
                    TRY_UNWRAP_WITH_DISCARD(tonemap_code,
                                            gpu.compiler->compile_from_file("assets/shaders/tonemap.slang",
                                                                            std::span(tonemap_names),
                                                                            std::span(tonemap_reflection)),
                                            "Failed to compile tonemap shader");

                    std::array<const std::string_view, 2> rotate_cubes_names{"rotate_geometry_cs", "rotate_lights_cs"};
                    std::array<ReflectionData, rotate_cubes_names.size()> rotate_cubes_reflection{};
                    TRY_UNWRAP_WITH_DISCARD(rotate_cubes_code,
                                            gpu.compiler->compile_from_file("assets/shaders/rotate_cubes.slang",
                                                                            std::span(rotate_cubes_names),
                                                                            std::span(rotate_cubes_reflection)),
                                            "Failed to compile rotate cubes shader");

                    std::array<const std::string_view, 3> gbuffer_entry_point_names = {"main_vs_mdi", "main_fs_mdi",
                                                                                       "fs_fullscreen_main"};
                    std::array<ReflectionData, gbuffer_entry_point_names.size()> gbuffer_reflection{};
                    TRY_UNWRAP_WITH_DISCARD(gbuffer_mrt_and_lighting_code,
                                            gpu.compiler->compile_from_file("assets/shaders/gbuffer.slang",
                                                                            std::span(gbuffer_entry_point_names),
                                                                            std::span(gbuffer_reflection)),
                                            "Failed to compile gbuffer shader");

                    std::array<const std::string_view, 1> present_names = {"present_fs"};
                    std::array<ReflectionData, present_names.size()> present_reflection{};
                    TRY_UNWRAP_WITH_DISCARD(present_code,
                                            gpu.compiler->compile_from_file("assets/shaders/present.slang",
                                                                            std::span(present_names),
                                                                            std::span(present_reflection)),
                                            "Failed to compile present shader");

                    std::array<const std::string_view, 1> debug_clustering_names = {"ClusterHeatmapCS"};
                    std::array<ReflectionData, debug_clustering_names.size()> debug_clustering_reflection{};
                    TRY_UNWRAP_WITH_DISCARD(debug_clustering_code,
                                            gpu.compiler->compile_from_file("assets/shaders/debug_clustering.slang",
                                                                            std::span(debug_clustering_names),
                                                                            std::span(debug_clustering_reflection)),
                                            "Failed to compile debug clustering shader");

                    std::array<const std::string_view, 3> debug_point_light_names = {"main_as", "main_ms",
                                                                                     "main_fs_debug"};
                    std::array<ReflectionData, debug_point_light_names.size()> debug_point_light_reflection{};
                    TRY_UNWRAP_WITH_DISCARD(debug_point_light_code,
                                            gpu.compiler->compile_from_file("assets/shaders/light_mesh.slang",
                                                                            std::span(debug_point_light_names),
                                                                            std::span(debug_point_light_reflection)),
                                            "Failed to compile point light mesh debug");

                    std::array<const std::string_view, 2> cubemap_names = {"cubemap_vs", "cubemap_fs"};
                    std::array<ReflectionData, cubemap_names.size()> cubemap_reflection{};
                    TRY_UNWRAP_WITH_DISCARD(cubemap_code,
                                            gpu.compiler->compile_from_file("assets/shaders/cubemap.slang",
                                                                            std::span(cubemap_names),
                                                                            std::span(cubemap_reflection)),
                                            "Failed to compile cubemap shader");

                    std::array<const std::string_view, 2> ssao_compute_names = {"ssao_compute", "ssao_blur"};
                    std::array<ReflectionData, ssao_compute_names.size()> ssao_compute_reflection{};
                    TRY_UNWRAP_WITH_DISCARD(ssao_compute_code,
                                            gpu.compiler->compile_from_file("assets/shaders/ssao_compute.slang",
                                                                            std::span(ssao_compute_names),
                                                                            std::span(ssao_compute_reflection)),
                                            "Failed to compile ssao compute shader");

                    std::array<const std::string_view, 3> bloom_names = {"bloom_threshold_cs", "bloom_downsample_cs",
                                                                         "bloom_upsample_cs"};
                    std::array<ReflectionData, bloom_names.size()> bloom_reflection{};
                    TRY_UNWRAP_WITH_DISCARD(bloom_code,
                                            gpu.compiler->compile_from_file("assets/shaders/bloom.slang",
                                                                            std::span(bloom_names),
                                                                            std::span(bloom_reflection)),
                                            "Failed to compile bloom shader");

                    std::array<const std::string_view, 2> point_light_billboard_names = {"billboard_ms",
                                                                                         "billboard_fs"};
                    std::array<ReflectionData, point_light_billboard_names.size()> point_light_billboard_reflection{};
                    TRY_UNWRAP_WITH_DISCARD(
                            point_light_billboard_code,
                            gpu.compiler->compile_from_file("assets/shaders/light_billboard.slang",
                                                            std::span(point_light_billboard_names),
                                                            std::span(point_light_billboard_reflection)),
                            "Failed to compile point light billboard shader");

                    auto &&[crp, lrp] = create_compute_pipelines(
                            gpu.device, gpu.ctx.pipeline_cache.get(), gpu.bindless.layout, sizeof(RotatePushConstant),
                            std::span(rotate_cubes_code), std::span(rotate_cubes_names));

                    auto &&[cl_groups, finalise_cl] = create_compute_pipelines(
                            gpu.device, gpu.ctx.pipeline_cache.get(), gpu.bindless.layout,
                            sizeof(ClusteredLightCullingPushConstants), std::span(clustered_culling_code),
                            std::span(clustered_culling_names));

                    auto gbuffer_pipeline = create_gbuffer_pipeline(
                            gpu.device, gpu.ctx.pipeline_cache.get(), gpu.bindless.layout,
                            gbuffer_mrt_and_lighting_code.at(0), gbuffer_mrt_and_lighting_code.at(1),
                            VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                            VK_FORMAT_D32_SFLOAT);

                    auto gbuf_light = create_deferred_lighting_graphics_pipeline(
                            gpu.device, gpu.ctx.pipeline_cache.get(), gpu.bindless.layout,
                            gbuffer_mrt_and_lighting_code.at(2), *gpu.ctx.shaders.get(pipes.fullscreen_vs),
                            "fs_fullscreen_main", VK_FORMAT_R16G16B16A16_SFLOAT);

                    auto pp = create_predepth_pipeline(gpu.device, gpu.ctx.pipeline_cache.get(), gpu.bindless.layout,
                                                       predepth_code.at(0), VK_FORMAT_D32_SFLOAT, gpu.msaa_samples);
                    auto pp_alpha = create_predepth_pipeline(
                            gpu.device, gpu.ctx.pipeline_cache.get(), gpu.bindless.layout, predepth_code.at(0),
                            predepth_code.at(1), VK_FORMAT_D32_SFLOAT, gpu.msaa_samples);

                    auto shadow_map_alpha = create_directional_shadow_map_pipeline(
                            gpu.device, gpu.ctx.pipeline_cache.get(), gpu.bindless.layout,
                            directional_shadow_map_code.at(0), directional_shadow_map_code.at(1), VK_FORMAT_D32_SFLOAT,
                            gpu.msaa_samples);
                    auto shadow_map = create_directional_shadow_map_pipeline(
                            gpu.device, gpu.ctx.pipeline_cache.get(), gpu.bindless.layout,
                            directional_shadow_map_code.at(0), VK_FORMAT_D32_SFLOAT, gpu.msaa_samples);

                    auto tp = create_fullscreen_pipeline(Pipeline::Fullscreen{
                            .device = gpu.device,
                            .cache = gpu.ctx.pipeline_cache.get(),
                            .bindless_layout = gpu.bindless.layout,
                            .fullscreen_vs = *gpu.ctx.shaders.get(pipes.fullscreen_vs),
                            .frag_code = tonemap_code.at(1),
                            .fs_entry = "fs_main",
                            .color_format = VK_FORMAT_R8G8B8A8_SRGB,
                            .push_constant_size = sizeof(TonemapPushConstants),
                            .enable_blend = false,
                    });

                    auto present_pipe = create_fullscreen_pipeline(Pipeline::Fullscreen{
                            .device = gpu.device,
                            .cache = gpu.ctx.pipeline_cache.get(),
                            .bindless_layout = gpu.bindless.layout,
                            .fullscreen_vs = *gpu.ctx.shaders.get(pipes.fullscreen_vs),
                            .frag_code = present_code.at(0),
                            .fs_entry = "present_fs",
                            .color_format = gpu.swapchain.format(),
                            .push_constant_size = sizeof(PresentPushConstants),
                            .enable_blend = false,
                    });

                    auto debug_pipeline = create_light_volume_mesh_pipeline(
                            gpu.device, gpu.ctx.pipeline_cache.get(), gpu.bindless.layout, debug_point_light_code.at(0),
                            debug_point_light_code.at(1), debug_point_light_code.at(2), VK_FORMAT_R16G16B16A16_SFLOAT,
                            VK_FORMAT_D32_SFLOAT, gpu.msaa_samples);

                    auto debug_clustering_pipeline = create_compute_pipeline(
                            gpu.device, gpu.ctx.pipeline_cache.get(), gpu.bindless.layout, debug_clustering_code.at(0),
                            sizeof(HeatmapPushConstants), "ClusterHeatmapCS");

                    {
                        const std::array skybox_stages{
                                Pipeline::ShaderStageInfo{cubemap_code.at(0), "cubemap_vs", VK_SHADER_STAGE_VERTEX_BIT},
                                Pipeline::ShaderStageInfo{cubemap_code.at(1), "cubemap_fs",
                                                          VK_SHADER_STAGE_FRAGMENT_BIT},
                        };
                        const std::array skybox_color_attachments{
                                Pipeline::ColorAttachmentInfo{.format = VK_FORMAT_R16G16B16A16_SFLOAT},
                        };
                        const std::array skybox_extra_dynamic_states{
                                VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
                                VK_DYNAMIC_STATE_DEPTH_BOUNDS,
                                VK_DYNAMIC_STATE_CULL_MODE,
                                VK_DYNAMIC_STATE_FRONT_FACE,
                        };
                        auto cubemap_pipeline = Pipeline::create_graphics_pipeline(Pipeline::Graphics{
                                .device = gpu.device,
                                .cache = gpu.ctx.pipeline_cache.get(),
                                .bindless_layout = gpu.bindless.layout,
                                .debug_name = "skybox",
                                .stages = skybox_stages,
                                .push_constant_size = sizeof(SkyboxPushConstants),
                                .push_constant_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                .color_attachments = skybox_color_attachments,
                                .depth_format = VK_FORMAT_D32_SFLOAT,
                                .depth_mode = Pipeline::DepthMode::test_greater_equal,
                                .cull_mode = Pipeline::CullMode::none,
                                .vertex_input = Pipeline::VertexInputInfo{},
                                .samples = gpu.msaa_samples,
                                .extra_dynamic_states = skybox_extra_dynamic_states,
                        });
                        hot_swap(pipes.skybox_pipeline, std::move(cubemap_pipeline), gpu.ctx, rc.retire_value);
                    }

                    auto ssao =
                            create_compute_pipeline(gpu.device, gpu.ctx.pipeline_cache.get(), gpu.bindless.layout,
                                                    ssao_compute_code.at(0), sizeof(SSAOPushConstants), "ssao_compute");
                    auto ssao_blur = create_compute_pipeline(gpu.device, gpu.ctx.pipeline_cache.get(),
                                                             gpu.bindless.layout, ssao_compute_code.at(1),
                                                             sizeof(SSAOBlurPushConstants), "ssao_blur");

                    auto bloom_threshold = create_compute_pipeline(
                            gpu.device, gpu.ctx.pipeline_cache.get(), gpu.bindless.layout, bloom_code.at(0),
                            sizeof(BloomThresholdPushConstants), "bloom_threshold_cs");

                    auto bloom_downsample = create_compute_pipeline(
                            gpu.device, gpu.ctx.pipeline_cache.get(), gpu.bindless.layout, bloom_code.at(1),
                            sizeof(BloomDownsamplePushConstants), "bloom_downsample_cs");

                    auto bloom_upsample = create_compute_pipeline(
                            gpu.device, gpu.ctx.pipeline_cache.get(), gpu.bindless.layout, bloom_code.at(2),
                            sizeof(BloomUpsamplePushConstants), "bloom_upsample_cs");

                    {
                        const std::array color_atts{Pipeline::ColorAttachmentInfo{
                                .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                                .blend_additive = false,
                        }};
                        auto pipe = Pipeline::create_mesh_pipeline(Pipeline::Mesh{
                                .device = gpu.device,
                                .cache = gpu.ctx.pipeline_cache.get(),
                                .bindless_layout = gpu.bindless.layout,
                                .debug_name = "billboard",
                                .stages =
                                        {
                                                .task = std::nullopt,
                                                .mesh = {point_light_billboard_code.at(0), "billboard_ms",
                                                         VK_SHADER_STAGE_MESH_BIT_EXT},
                                                .fragment = {point_light_billboard_code.at(1), "billboard_fs",
                                                             VK_SHADER_STAGE_FRAGMENT_BIT},
                                        },
                                .push_constant_size = sizeof(BillboardPushConstants),
                                .push_constant_stages = VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                .color_attachments = color_atts,
                                .depth_format = VK_FORMAT_D32_SFLOAT, // depth_format,
                                .depth_mode = Pipeline::DepthMode::test_greater_equal,
                                .cull_mode = Pipeline::CullMode::none,
                        });

                        hot_swap(pipes.billboard_pipeline, std::move(pipe), gpu.ctx, rc.retire_value);
                    }

                    hot_swap(pipes.bloom_threshold_pipeline, std::move(bloom_threshold), gpu.ctx, rc.retire_value);
                    hot_swap(pipes.bloom_downsample_pipeline, std::move(bloom_downsample), gpu.ctx, rc.retire_value);
                    hot_swap(pipes.bloom_upsample_pipeline, std::move(bloom_upsample), gpu.ctx, rc.retire_value);

                    hot_swap(pipes.gbuffer_pipeline_lighting, std::move(gbuf_light), gpu.ctx, rc.retire_value);
                    hot_swap(pipes.cube_rotation_pipeline, std::move(crp), gpu.ctx, rc.retire_value);
                    hot_swap(pipes.light_rotation_pipeline, std::move(lrp), gpu.ctx, rc.retire_value);
                    hot_swap(pipes.gbuffer_pipeline_mrt, std::move(gbuffer_pipeline), gpu.ctx, rc.retire_value);
                    hot_swap(pipes.finalise_compact_pipeline, std::move(finalise_cl), gpu.ctx, rc.retire_value);
                    hot_swap(pipes.predepth_pipeline, std::move(pp), gpu.ctx, rc.retire_value);
                    hot_swap(pipes.predepth_alpha_pipeline, std::move(pp_alpha), gpu.ctx, rc.retire_value);
                    hot_swap(pipes.tonemap_pipeline, std::move(tp), gpu.ctx, rc.retire_value);
                    hot_swap(pipes.cluster_build_groups_pipeline, std::move(cl_groups), gpu.ctx, rc.retire_value);
                    hot_swap(pipes.present_pipeline, std::move(present_pipe), gpu.ctx, rc.retire_value);
                    hot_swap(pipes.directional_shadow_map_pipeline, std::move(shadow_map), gpu.ctx, rc.retire_value);
                    hot_swap(pipes.directional_shadow_map_alpha_pipeline, std::move(shadow_map_alpha), gpu.ctx,
                             rc.retire_value);
                    hot_swap(pipes.debug_point_light_pipeline, std::move(debug_pipeline), gpu.ctx, rc.retire_value);
                    hot_swap(pipes.debug_light_clustering, std::move(debug_clustering_pipeline), gpu.ctx,
                             rc.retire_value);
                    hot_swap(pipes.ssao_pipeline, std::move(ssao), gpu.ctx, rc.retire_value);
                    hot_swap(pipes.ssao_blur_pipeline, std::move(ssao_blur), gpu.ctx, rc.retire_value);
                },
                ResizeTrigger::Shaders);
    }

    {
        const auto offscreen_node = gpu.scene_resize_graph.add_node("offscreen_targets", [&](VkExtent2D e,
                                                                                             const ResizeContext &rc) {
            hot_swap(res.gbuffer0,
                     create_offscreen_target(gpu.allocator, e.width, e.height, VK_FORMAT_R8G8B8A8_UNORM, {},
                                             "gbuffer0_albedo_ao"),
                     gpu.ctx, rc.retire_value);

            hot_swap(res.gbuffer1,
                     create_offscreen_target(gpu.allocator, e.width, e.height, VK_FORMAT_R16G16B16A16_SFLOAT, {},
                                             "gbuffer1_normal_rough_metal"),
                     gpu.ctx, rc.retire_value);

            hot_swap(res.gbuffer2,
                     create_offscreen_target(gpu.allocator, e.width, e.height, VK_FORMAT_R16G16B16A16_SFLOAT, {},
                                             "gbuffer2_emissive"),
                     gpu.ctx, rc.retire_value);

            const u32 cell_size = 16;
            const u32 slices_per_row = 4;
            const u32 hm_w = res.clustering_config.tiles_x * slices_per_row * cell_size;
            const u32 hm_h =
                    res.clustering_config.tiles_y * (res.clustering_config.tiles_z / slices_per_row) * cell_size;
            hot_swap(res.debug_culling,
                     create_offscreen_target(gpu.allocator, hm_w, hm_h, VK_FORMAT_R16G16B16A16_SFLOAT, {},
                                             "debug_culling"),
                     gpu.ctx, rc.retire_value);

            // SSAO output
            hot_swap(res.ssao_output,
                     create_offscreen_target(gpu.allocator, e.width, e.height, VK_FORMAT_R8_UNORM, {}, "ssao_output"),
                     gpu.ctx, rc.retire_value);
            // SSAO blur
            hot_swap(res.ssao_blurred,
                     create_offscreen_target(gpu.allocator, e.width, e.height, VK_FORMAT_R8_UNORM, {}, "ssao_blurred"),
                     gpu.ctx, rc.retire_value);
            hot_swap(
                    res.ssao_blurred_temp,
                    create_offscreen_target(gpu.allocator, e.width, e.height, VK_FORMAT_R8_UNORM, {}, "ssao_blur_temp"),
                    gpu.ctx, rc.retire_value);

            hot_swap(res.depth,
                     create_depth_target(gpu.allocator, e.width, e.height, VK_FORMAT_D32_SFLOAT, VK_SAMPLE_COUNT_1_BIT,
                                         true, "depth"),
                     gpu.ctx, rc.retire_value);

            if (res.directional_shadow_map_depth.empty()) {
                res.directional_shadow_map_depth = gpu.ctx.create_texture(create_depth_target(
                        gpu.allocator, ui.shadow_map_resolution.peek(), ui.shadow_map_resolution.peek(),
                        VK_FORMAT_D32_SFLOAT, VK_SAMPLE_COUNT_1_BIT, true, "directional_shadow_map"));
            }

            hot_swap(res.lit_hdr,
                     create_offscreen_target(gpu.allocator, e.width, e.height, VK_FORMAT_R16G16B16A16_SFLOAT, {},
                                             "lit_hdr"),
                     gpu.ctx, rc.retire_value);
        });

        const auto tonemapped_node = gpu.scene_resize_graph.add_node(
                "tonemapped_image", [&](VkExtent2D e, const ResizeContext &resize_context) {
                    const auto old_tonemap = res.tonemapped;

                    res.tonemapped = gpu.ctx.create_texture(create_offscreen_target(
                            gpu.allocator, e.width, e.height, VK_FORMAT_R8G8B8A8_SRGB, {}, "tonemapped"));
                    destroy(gpu.ctx, old_tonemap, resize_context.retire_value);
                });

        std::ignore = gpu.scene_resize_graph.add_node(
                "directional_shadow_map",
                [&](VkExtent2D, const ResizeContext &rc) {
                    auto maybe_new_res = ui.shadow_map_resolution.consume_if_changed();
                    if (!maybe_new_res)
                        return;

                    const u32 sm_res = *maybe_new_res;

                    const auto old_sm = res.directional_shadow_map_depth;

                    res.directional_shadow_map_depth = gpu.ctx.create_texture(
                            create_depth_target(gpu.allocator, sm_res, sm_res, VK_FORMAT_D32_SFLOAT,
                                                VK_SAMPLE_COUNT_1_BIT, true, "directional_shadow_map"));

                    destroy(gpu.ctx, old_sm, rc.retire_value);
                },
                ResizeTrigger::ShadowMap);

        std::ignore = gpu.scene_resize_graph.add_node(
                "clustering_update",
                [&](VkExtent2D, const ResizeContext &rc) {
                    auto maybe_cfg = ui.clustering_config.consume_if_changed();
                    if (!maybe_cfg)
                        return;

                    const auto &cfg = *maybe_cfg;

                    // Update the central config
                    res.clustering_config =
                            cluster_config(cfg.tiles_x, cfg.tiles_y, cfg.tiles_z, cfg.z_near, cfg.z_far);
                    res.max_light_indices = res.clustering_config.cluster_count * max_lights_per_cluster;

                    // Use the new recreation pattern
                    AlignedRingBuffer<Cluster>::recreate(gpu.ctx, rc.retire_value, res.clusters,
                                                         res.clustering_config.cluster_count, "clusters_ring");

                    AlignedRingBuffer<u32>::recreate(gpu.ctx, rc.retire_value, res.cluster_light_indices,
                                                     res.max_light_indices, "cluster_light_indices_ring");

                    const u32 cell_size = 16;
                    const u32 slices_per_row = 4;
                    const u32 hm_w = res.clustering_config.tiles_x * slices_per_row * cell_size;
                    const u32 hm_h = res.clustering_config.tiles_y * (res.clustering_config.tiles_z / slices_per_row) *
                                     cell_size;
                    hot_swap(res.debug_culling,
                             create_offscreen_target(gpu.allocator, hm_w, hm_h, VK_FORMAT_R16G16B16A16_SFLOAT, {},
                                                     "debug_culling"),
                             gpu.ctx, rc.retire_value);
                },
                ResizeTrigger::Clustering);

        const auto bloom_node =
                gpu.scene_resize_graph.add_node("bloom_targets", [&](VkExtent2D e, const ResizeContext &rc) {
                    hot_swap(res.bloom_threshold,
                             create_offscreen_target(gpu.allocator, e.width, e.height, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                     {}, "bloom_threshold"),
                             gpu.ctx, rc.retire_value);

                    // Resize the vectors if mip count changed or first run
                    const u32 mip_count = res.bloom_mip_count;

                    if (res.bloom_downsample.size() != mip_count) {
                        for (auto h: res.bloom_downsample)
                            destroy(gpu.ctx, h, rc.retire_value);
                        for (auto h: res.bloom_upsample)
                            destroy(gpu.ctx, h, rc.retire_value);
                        res.bloom_downsample.resize(mip_count);
                        res.bloom_upsample.resize(mip_count - 1);
                    }

                    for (u32 i = 0; i < mip_count; ++i) {
                        const u32 w = std::max(1u, e.width >> (i + 1));
                        const u32 h = std::max(1u, e.height >> (i + 1));

                        hot_swap(res.bloom_downsample[i],
                                 create_offscreen_target(gpu.allocator, w, h, VK_FORMAT_R16G16B16A16_SFLOAT, {},
                                                         std::format("bloom_ds_{}", i)),
                                 gpu.ctx, rc.retire_value);

                        if (i < mip_count - 1) {
                            hot_swap(res.bloom_upsample[i],
                                     create_offscreen_target(gpu.allocator, w, h, VK_FORMAT_R16G16B16A16_SFLOAT, {},
                                                             std::format("bloom_us_{}", i)),
                                     gpu.ctx, rc.retire_value);
                        }
                    }
                });

        gpu.scene_resize_graph.add_dependency(bloom_node, offscreen_node);
        gpu.scene_resize_graph.add_dependency(tonemapped_node, bloom_node);

        gpu.scene_resize_graph.add_dependency(tonemapped_node, offscreen_node);
    }

    /*{
        res.environment_cubemap =
    generate_sky_cubemap(gpu.ctx, gpu.allocator, gpu.device,
                         gpu.bindless.layout, gpu.bindless.set,
                         *gpu.ctx.command_ctx, *gpu.compiler,
                         glm::vec3(0.0f, 0.3f, -1.0f), 22.0f, 512).value();
gpu.bindless.need_repopulate = true;
    }*/
    {
        res.environment_cubemap = gpu.ctx.create_texture_owned(
                load_cubemap_ktx(gpu.allocator, *gpu.ctx.command_ctx, gpu.device, gpu.physical_device,
                                 gpu.graphics_queue, std::filesystem::path("assets/editor/cubemaps/nasa/sky.ktx2"),
                                 "environment_cubemap")
                        .value());
    }


    auto last_window_extent = sanitize_window_extent(current_extent(gpu.window), gpu.physical_device, gpu.surface);
    auto last_scene_extent = VkExtent2D{opts.width, opts.height};

    gpu.window_resize_graph.rebuild(last_window_extent, ResizeContext{.ctx = gpu.ctx, .retire_value = 0},
                                    ResizeTrigger::Extent);
    gpu.scene_resize_graph.rebuild(last_scene_extent, ResizeContext{.ctx = gpu.ctx, .retire_value = 0},
                                   ResizeTrigger::Extent);

    ui.last_viewport_extent = last_scene_extent;

    {
        NANO_SCOPE("Create ImGuiRenderer");
        ui.gui = std::make_unique<ImGuiRenderer>(
                gpu.window, static_cast<u32>(gpu.swapchain.image_count()), gpu.ctx, *gpu.compiler,
                FontChoice{
                        .font_path = "assets/editor/fonts/IBM_Plex_Mono/IBMPlexMono-Regular.ttf",
                        .size = 13.0f,
                });
        ui.gui->set_app_name("BHEngine");
    }

    auto gui_pipeline_node = gpu.window_resize_graph.add_node(
            "gui_pipeline", [&gui = *ui.gui](auto, const auto &) { gui.set_should_recompile(); },
            ResizeTrigger::Shaders);
    gpu.window_resize_graph.add_dependency(gui_pipeline_node, pipelines_node);

    {
        NANO_SCOPE("Create file watcher");
        ui.watcher = std::unique_ptr<efsw::FileWatcher, Deleter>(new efsw::FileWatcher(false), Deleter{});
        ui.listeners["update"] = std::unique_ptr<efsw::FileWatchListener, Deleter>(
                new ShaderSourceCodeChangeListener(&gpu.window_resize_graph), Deleter{});
        std::ignore = ui.watcher->addWatch("assets/shaders", ui.listeners["update"].get(), true,
                                           {efsw::WatcherOption(efsw::Option::WinBufferSize, 128 * 1024)});
        ui.watcher->watch();
    }

    // --- Graph setup ---
    if (!ui.graphs_initialized) {
        for (auto idx: compute_stages)
            ui.gpu_frame_graph.add_line(get_compute_pass_name(idx));
        for (auto idx: graphics_stages)
            ui.gpu_frame_graph.add_line(get_graphics_pass_name(idx));
        ui.graphs_initialized = true;
    }
    ui.last_frame_time = std::chrono::high_resolution_clock::now();
    auto stats = FrameStats{};

    {
        NANO_SCOPE("Show window");
        glfwShowWindow(gpu.window);
        glfwFocusWindow(gpu.window);
    }

    // Precompute device addresses used in push constants
    const auto point_lights_base_addr = gpu.ctx.device_address(res.point_lights_base);

    while (!glfwWindowShouldClose(gpu.window) && keep_running) {
        if (!keep_running) {
            res.asset_streamer->emergency_shutdown();
            break;
        }

        glfwPollEvents();
        poll_streamer(res, gpu);
        handle_bindless_repopulation(app_context, gpu.window_resize_graph);


        update_frame_timing(ui);
        const auto [bounded_frame_index, last_frame_index] = frame_indices(ui);

        auto ui_frame = run_ui_frame(app_context);
        if (ui_frame.minimized) {
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
            continue;
        }

        VkExtent2D render_scene_extent = choose_render_scene_extent(ui, ui_frame.desired_scene_extent);

        update_pending_resize(ui, ui_frame.desired_scene_extent);

        if (commit_resizes(app_context, gpu.window_resize_graph, gpu.scene_resize_graph, ui_frame.window_extent,
                           last_window_extent, render_scene_extent)) {
            continue;
        }

        if (ui.capture_next_frame) {
            renderdoc->begin_frame_capture(instance.instance);
            ui.capture_next_frame = false;
        }


        res.draw_stream.begin_frame();

        // camera update + frame ubo write
        poll_gamepad(ui.app_state);
        ui.app_state.cam.update(gpu.window, ui.dt, ui.app_state.cam_in);

        write_camera_to_frame_ubo(*res.frame_ubo, gpu.ctx, res.frame_ubo_ring, bounded_frame_index, ui.app_state.cam,
                                  render_scene_extent, fov_y, z_near, z_far);

        ui.total_time += ui.dt;
        {
            const float elevation = glm::radians(ui.sun_config.elevation_degrees);
            const float azimuth = glm::radians(ui.sun_config.azimuth_degrees);

            const auto sun_dir = glm::normalize(glm::vec3{
                    std::sin(azimuth) * std::cos(elevation),
                    std::sin(elevation),
                    std::cos(azimuth) * std::cos(elevation),
            });

            auto sun_direction_intensity = glm::vec4(sun_dir, ui.sun_config.intensity);
            auto offset = offsetof(FrameUBO, sun_direction_intensity);
            res.frame_ubo_ring.write_field(gpu.ctx, bounded_frame_index, sun_direction_intensity, offset);
            /*{
                const glm::vec3 light_pos =
                        ui.shadow_config.light_target - (sun_dir * ui.shadow_config.shadow_distance);
                glm::mat4 light_view =
                        glm::lookAt(light_pos, ui.shadow_config.light_target, glm::vec3(0.0f, 1.0f, 0.0f));
                const float ortho_size = ui.shadow_config.ortho_size;

                const float half_size = ortho_size * 0.5F;

                const glm::mat4 light_proj = glm::orthoRH_ZO(-half_size, half_size, -half_size, half_size,
            ui.shadow_config.far_plane, ui.shadow_config.near_plane); ui.shadow_config.light_view_proj =  light_proj *
            light_view;
            }*/
            {
                glm::vec3 scene_center = res.meshes.at(0).mesh_aabb.center();
                glm::vec3 scene_extents = (res.meshes.at(0).mesh_aabb.max - res.meshes.at(0).mesh_aabb.min) * 0.5f;
                float scene_radius = glm::length(scene_extents);
                float light_distance = scene_radius * 2.0f + 50.0f;
                glm::vec3 light_pos = scene_center - sun_dir * light_distance;

                constexpr auto up_vector = glm::vec3(0.0f, 1.0f, 0.0f);
                glm::mat4 light_view = glm::lookAt(light_pos, scene_center, up_vector);

                // Transform AABB to light space
                auto [ls_min, ls_max] = res.meshes.at(0).mesh_aabb.transform(light_view);
                // Calculate bounds with padding
                const float padding = 20.0f;
                float ortho_size = glm::max(ls_max.x - ls_min.x, ls_max.y - ls_min.y) + padding;
                float half_size = ortho_size * 0.5f;
                // Near/far planes
                float near_plane = glm::max(-ls_max.z - padding, 0.1f);
                float far_plane = -ls_min.z + padding;
                const glm::mat4 light_proj =
                        glm::orthoRH_ZO(-half_size, half_size, -half_size, half_size, near_plane, far_plane);
                ui.shadow_config.light_view_proj = light_proj * light_view;
            }
        }

        submit_mesh_instances(scene.scene, scene.render_queue);
        flush_render_queue(scene.render_queue, res, gpu.ctx, bounded_frame_index);
        flush_material_pool(gpu.ctx);

        IndirectWriteBuffers write_buffers{
                .writer = res.draw_stream.writer,
                .cmd_ring = res.indirect_ring,
                .material_id_ring = res.draw_material_id_ring,
        };
        std::vector<DrawRanges> all_mesh_ranges;
        all_mesh_ranges.reserve(res.mesh_instance_ranges.size());
        for (const auto &mir: res.mesh_instance_ranges) {
            auto &mesh = res.meshes.at(mir.mesh_index);
            if (mir.instance_count == 0)
                continue;
            all_mesh_ranges.push_back(write_mesh_indirect(gpu.ctx, bounded_frame_index, write_buffers, mesh.mesh,
                                                          MeshDrawInfo{
                                                                  .mesh_index = mir.mesh_index,
                                                                  .material_pool_base = mesh.material_pool_base,
                                                                  .instance_count = mir.instance_count,
                                                                  .first_instance = mir.base_instance,
                                                                  .overrides = res.submesh_material_overrides,
                                                          }));
        }

        const u32 light_slot = reserve_light_volumes(gpu.ctx, bounded_frame_index, res.draw_stream.writer,
                                                     res.mesh_indirect_ring, res.draw_material_id_ring, 0u);

        auto &fs = res.frames[bounded_frame_index];

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

            if (auto r = read_timestamp_pairs_ms(gpu.ctx, pipes.compute_query_pool[bounded_frame_index]))
                ui.last_compute_res.update(std::move(*r));

            if (auto r = read_timestamp_pairs_ms(gpu.ctx, pipes.graphics_query_pool[bounded_frame_index]))
                ui.last_graphics_res.update(std::move(*r));

            if (auto r = read_graphics_stats(gpu.ctx, pipes.graphics_stats_pool[bounded_frame_index]))
                ui.last_g_stats.update(std::move(*r));

            if (auto r = read_compute_stats(gpu.ctx, pipes.compute_stats_pool[bounded_frame_index]))
                ui.last_c_stats.update(std::move(*r));

            auto &&[a, b, c, d] = gpu.ctx.query_pools.get_multiple(
                    pipes.compute_query_pool[bounded_frame_index], pipes.graphics_query_pool[bounded_frame_index],
                    pipes.graphics_stats_pool[bounded_frame_index], pipes.compute_stats_pool[bounded_frame_index]);

            vkResetQueryPool(gpu.device, a->pool, 0, a->query_count);
            vkResetQueryPool(gpu.device, b->pool, 0, b->query_count);
            vkResetQueryPool(gpu.device, c->pool, 0, c->query_count);
            vkResetQueryPool(gpu.device, d->pool, 0, d->query_count);

            // This causes validation errors.
            TracyVkCollectHost(gpu.tracy_compute.ctx);
            TracyVkCollectHost(gpu.tracy_graphics.ctx);
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

        RP::setup_render_passes_for_frame(app_context, bounded_frame_index);


        {
            fs.timeline_values[stage_index(Stage::CubeRotation)] =
                    run_rotation_pass(app_context, bounded_frame_index, last_frame_index, point_lights_base_addr);
        }

        {
            const std::array cube_rotate_waits{TimelineWait{
                    .value = fs.timeline_values[stage_index(Stage::CubeRotation)],
                    .semaphore = gpu.tl_compute.timeline,
                    .stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
            }};
            fs.timeline_values[stage_index(Stage::Predepth)] =
                    run_predepth_pass(app_context, render_scene_extent, res.mesh_instance_ranges, all_mesh_ranges,
                                      bounded_frame_index, SubmitSynchronisation{.timeline_waits = cube_rotate_waits});
        }

        {
            const std::array directional_shadow_map_waits{TimelineWait{
                    .value = fs.timeline_values[stage_index(Stage::CubeRotation)],
                    .semaphore = gpu.tl_compute.timeline,
                    .stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
            }};
            fs.timeline_values[stage_index(Stage::DirectionalShadowMap)] = run_directional_shadow_map_pass(
                    app_context, res.mesh_instance_ranges, all_mesh_ranges, bounded_frame_index,
                    SubmitSynchronisation{.timeline_waits = directional_shadow_map_waits});
        }

        {
            fs.timeline_values[stage_index(Stage::LightClustering)] =
                    run_light_clustering_pass(app_context, bounded_frame_index, no_waits);
        }

        {
            const std::array gbuffer_waits{
                    TimelineWait{
                            .value = fs.timeline_values[stage_index(Stage::CubeRotation)],
                            .semaphore = gpu.tl_compute.timeline,
                            .stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                    },
                    TimelineWait{
                            .value = fs.timeline_values[stage_index(Stage::Predepth)],
                            .semaphore = gpu.tl_graphics.timeline,
                            .stage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                    },
                    TimelineWait{
                            .value = fs.timeline_values[stage_index(Stage::DirectionalShadowMap)],
                            .semaphore = gpu.tl_graphics.timeline,
                            .stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    },
            };
            fs.timeline_values[stage_index(Stage::GBuffer)] =
                    run_gbuffer_pass(app_context, render_scene_extent, res.mesh_instance_ranges, all_mesh_ranges,
                                     bounded_frame_index, SubmitSynchronisation{.timeline_waits = gbuffer_waits});
        }

        {
            const std::array ssao_waits{
                    TimelineWait{
                            .value = fs.timeline_values[stage_index(Stage::GBuffer)],
                            .semaphore = gpu.tl_graphics.timeline,
                            .stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    },
            };
            fs.timeline_values[stage_index(Stage::SSAO)] =
                    run_ssao_pass(app_context, render_scene_extent, bounded_frame_index,
                                  SubmitSynchronisation{.timeline_waits = ssao_waits});
        }

        {
            const std::array ssao_blur_waits{
                    TimelineWait{
                            .value = fs.timeline_values[stage_index(Stage::SSAO)],
                            .semaphore = gpu.tl_compute.timeline,
                            .stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    },
            };
            fs.timeline_values[stage_index(Stage::SSAOBlur)] = run_ssao_blur_pass(
                    app_context, render_scene_extent, SubmitSynchronisation{.timeline_waits = ssao_blur_waits});
        }

        {
            const std::array deferred_waits{
                    TimelineWait{
                            .value = fs.timeline_values[stage_index(Stage::GBuffer)],
                            .semaphore = gpu.tl_graphics.timeline,
                            .stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    },
                    TimelineWait{
                            .value = fs.timeline_values[stage_index(Stage::SSAOBlur)],
                            .semaphore = gpu.tl_compute.timeline,
                            .stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    },
                    TimelineWait{
                            .value = fs.timeline_values[stage_index(Stage::LightClustering)],
                            .semaphore = gpu.tl_compute.timeline,
                            .stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    },
            };
            fs.timeline_values[stage_index(Stage::DeferredLighting)] =
                    run_deferred_lighting_pass(app_context, render_scene_extent, light_slot, bounded_frame_index,
                                               SubmitSynchronisation{.timeline_waits = deferred_waits});
        }

        {
            const std::array skybox_waits{
                    TimelineWait{
                            .value = fs.timeline_values[stage_index(Stage::DeferredLighting)],
                            .semaphore = gpu.tl_graphics.timeline,
                            .stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    },
            };
            fs.timeline_values[stage_index(Stage::Skybox)] =
                    run_environment_skybox_pass(app_context, render_scene_extent, bounded_frame_index,
                                                SubmitSynchronisation{.timeline_waits = skybox_waits});
        }

        {
            const std::array bloom_waits{
                    TimelineWait{
                            .value = fs.timeline_values[stage_index(Stage::Skybox)],
                            .semaphore = gpu.tl_graphics.timeline,
                            .stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    },
            };
            fs.timeline_values[stage_index(Stage::Bloom)] = run_bloom_pass(
                    app_context, render_scene_extent, SubmitSynchronisation{.timeline_waits = bloom_waits});
        }

        {
            const std::array billboard_waits{
                    TimelineWait{
                            .value = fs.timeline_values[stage_index(Stage::Bloom)],
                            .semaphore = gpu.tl_compute.timeline,
                            .stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    },
            };
            fs.timeline_values[stage_index(Stage::Billboard)] =
                    run_billboard_pass(app_context, render_scene_extent, bounded_frame_index,
                                       SubmitSynchronisation{.timeline_waits = billboard_waits});
        }

        {
            const std::array tonemap_waits{
                    TimelineWait{
                            .value = fs.timeline_values[stage_index(Stage::Billboard)],
                            .semaphore = gpu.tl_graphics.timeline,
                            .stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    },
            };
            fs.timeline_values[stage_index(Stage::Tonemapping)] =
                    run_tonemap_pass(app_context, render_scene_extent, bounded_frame_index,
                                     SubmitSynchronisation{.timeline_waits = tonemap_waits});
        }

        {
            const std::array present_timeline_waits{TimelineWait{
                    .value = fs.timeline_values[stage_index(Stage::Tonemapping)],
                    .semaphore = gpu.tl_graphics.timeline,
                    .stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            }};

            const std::array present_binary_waits{BinaryWait{
                    .semaphore = frame_sync.image_available,
                    .stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            }};

            const std::array present_binary_signals{BinarySignal{
                    .semaphore = frame_sync.render_finished,
                    .stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            }};

            fs.frame_done_value = run_swapchain_pass(app_context, swap_image_index, bounded_frame_index,
                                                     SubmitSynchronisation{
                                                             .binary_waits = present_binary_waits,
                                                             .timeline_waits = present_timeline_waits,
                                                             .binary_signals = present_binary_signals,
                                                     });
        }

        const auto completed = std::min(gpu.tl_compute.completed, gpu.tl_graphics.completed);
        gpu.ctx.destroy_queue.retire(completed);

        auto frame_end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(frame_end - ui.last_frame_time)
                          .count();
        stats.add_sample(ms);

        const VkResult present_res =
                gpu.swapchain.present(gpu.graphics_queue, swap_image_index, frame_sync.render_finished);
        if (present_res == VK_ERROR_OUT_OF_DATE_KHR || present_res == VK_SUBOPTIMAL_KHR) {
            auto result = gpu.swapchain.recreate(current_extent(gpu.window));
            if (!result)
                vk_check(result.error());
        } else {
            vk_check(present_res);
        }

        renderdoc->end_frame_capture(instance.instance);

        FrameMark;
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

    res.asset_streamer.reset();
    ui.gui.reset();
    gpu.ctx.clear_all();

    gpu.compiler.reset();

    ui.watcher.reset();
    ui.listeners.clear();

    gpu.ctx.destroy_queue.retire(std::numeric_limits<u64>::max());

    destruction::global_command_context(*gpu.ctx.command_ctx);
    destruction::bindless_set(gpu.bindless);
    destruction::timelines(gpu.device, gpu.tl_graphics, gpu.tl_transfer, gpu.tl_compute);
    destruction::allocator(gpu.allocator);
    destruction::swapchain(gpu.swapchain);
    destruction::wsi(instance.instance, gpu.surface, gpu.window);


    gpu.tracy_compute.shutdown();
    gpu.tracy_graphics.shutdown();

    destruction::device(gpu.device);
    destruction::instance(instance);
    volkFinalize();
    glfwTerminate();

    return 0;
}
