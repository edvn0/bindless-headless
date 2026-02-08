#include "app/frame.hxx"

auto fill_frame_ubo_from_camera(FrameUBO &ubo, const EditorCamera &cam, VkExtent2D extent, float fov_y_radians,
                                       float z_near) -> void {
    const float aspect = static_cast<float>(extent.width) / std::max(1.0f, static_cast<float>(extent.height));

    ubo.view = cam.view_matrix();
    ubo.projection = PerspectiveRH_ReverseZ_Inf(fov_y_radians, aspect, z_near);
    ubo.inv_projection = glm::inverse(ubo.projection);
    ubo.view_projection = ubo.projection * ubo.view;
    ubo.camera_position = glm::vec4(cam.camera_position(), 1.0f);
    ubo.inv_view_projection = glm::inverse(ubo.view_projection);
    ubo.viewport_size = glm::vec2(static_cast<float>(extent.width), static_cast<float>(extent.height));

    const auto normal_projection = glm::inverse(glm::perspective(fov_y_radians, aspect, 0.1F, 1000.0F));
    const auto planes = extract_frustum_planes(normal_projection);
    ubo.frustum_planes = {planes[0], planes[1], planes[2], planes[3], planes[4], planes[5]};
}

 auto write_camera_to_frame_ubo(RenderContext &ctx, AlignedRingBuffer<FrameUBO> &frame_ubo_ring, u32 frame_index,
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
