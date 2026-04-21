#pragma once

#include "Forward.hxx"
#include "app/app_forward.hxx"
#include "app/math.hxx"


#include "AlignedRingBuffer.hxx"
#include "Camera.hxx"

#include <glm/glm.hpp>

struct FrameUBO {
    glm::mat4 view{};
    glm::mat4 projection{};
    glm::mat4 view_projection{};
    glm::mat4 inv_projection{};
    glm::mat4 inv_view_projection{};
    glm::mat4 inv_view_projection_no_translation{};
    glm::vec4 camera_position{};
    std::array<FrustumPlane, 6> frustum_planes{}; // left, right, bottom, top, near, far
    glm::vec4 sun_direction_intensity{};
    glm::vec2 viewport_size{};
    u32 frame_index{};
};


auto fill_frame_ubo_from_camera(FrameUBO &ubo, const EditorCamera &cam, VkExtent2D extent, float fov_y_radians,
                                float z_near, float z_far) -> void;
auto write_camera_to_frame_ubo(FrameUBO &, RenderContext &ctx, AlignedRingBuffer<FrameUBO> &frame_ubo_ring,
                               u32 frame_index, const EditorCamera &cam, VkExtent2D extent, float fov_y_radians,
                               float z_near, float z_far) -> void;


struct ExtentBounds {
    u32 min_dim{1};
    u32 max_dim{4096};
};

auto clamp_u32(u32 v, u32 lo, u32 hi) -> u32;
auto sanitize_window_extent(VkExtent2D raw, VkPhysicalDevice physical_device, VkSurfaceKHR surface,
                            ExtentBounds bounds = {}) -> VkExtent2D;
auto sanitize_scene_extent(VkExtent2D raw, VkExtent2D fallback_last_valid, VkPhysicalDevice physical_device,
                           ExtentBounds bounds = {}) -> VkExtent2D;

auto update_frame_timing(AppUI &) -> void;
auto frame_indices(AppUI const &) -> std::pair<u32, u32>;
auto handle_bindless_repopulation(AppContext &, ResizeGraph &) -> void;

auto update_pending_resize(AppUI &, VkExtent2D) -> void;
auto commit_resizes(AppContext &, ResizeGraph &, ResizeGraph &, VkExtent2D, VkExtent2D &, VkExtent2D &) -> bool;
auto choose_render_scene_extent(AppUI const &, VkExtent2D) -> VkExtent2D;
