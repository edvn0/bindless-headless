#pragma once

#include "app/math.hxx"

#include "Camera.hxx"
#include "AlignedRingBuffer.hxx"

#include <glm/glm.hpp>

struct FrameUBO {
    glm::mat4 view;
    glm::mat4 projection;
    glm::mat4 view_projection;
    glm::mat4 inv_projection;
    glm::mat4 inv_view_projection;
    glm::vec4 camera_position;
    std::array<FrustumPlane, 6> frustum_planes; // left, right, bottom, top, near, far
    glm::vec4 sun_direction_intensity;
    glm::vec2 viewport_size;
};


 auto fill_frame_ubo_from_camera(FrameUBO &ubo, const EditorCamera &cam, VkExtent2D extent, float fov_y_radians,
                                       float z_near) -> void;
 auto write_camera_to_frame_ubo(RenderContext &ctx, AlignedRingBuffer<FrameUBO> &frame_ubo_ring, u32 frame_index,
                                      const EditorCamera &cam, VkExtent2D extent, float fov_y_radians, float z_near)
        -> void;
