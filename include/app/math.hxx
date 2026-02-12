#pragma once

#include "Allocator.hxx"
#include "Forward.hxx"
#include "Types.hxx"

constexpr auto msaa_from_cli = [](u32 v) -> VkSampleCountFlagBits {
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

constexpr auto current_extent = [](GLFWwindow *win) {
    int fbw{0};
    int fbh{0};
    glfwGetFramebufferSize(win, &fbw, &fbh);
    return VkExtent2D{.width = static_cast<u32>(std::max(fbw, 0)), .height = static_cast<u32>(std::max(fbh, 0))};
};

constexpr auto clamp_msaa_samples = [](VkPhysicalDevice physical_device,
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
struct FrustumPlane {
    glm::vec4 plane; // xyz = normal, w = distance
};


auto PerspectiveRH_ReverseZ_Inf(float fovYRadians, float aspect, float zNear) -> glm::mat4;
auto OrthoRH_ReverseZ(f32, f32, f32, f32, f32, f32) -> glm::mat4;

constexpr auto extract_frustum_planes = [](const glm::mat4 &view_proj) -> std::array<FrustumPlane, 6> {
    std::array<FrustumPlane, 6> planes{};

    planes[0].plane = glm::vec4(view_proj[0][3] + view_proj[0][0], view_proj[1][3] + view_proj[1][0],
                                view_proj[2][3] + view_proj[2][0], view_proj[3][3] + view_proj[3][0]);

    planes[1].plane = glm::vec4(view_proj[0][3] - view_proj[0][0], view_proj[1][3] - view_proj[1][0],
                                view_proj[2][3] - view_proj[2][0], view_proj[3][3] - view_proj[3][0]);

    planes[2].plane = glm::vec4(view_proj[0][3] + view_proj[0][1], view_proj[1][3] + view_proj[1][1],
                                view_proj[2][3] + view_proj[2][1], view_proj[3][3] + view_proj[3][1]);

    planes[3].plane = glm::vec4(view_proj[0][3] - view_proj[0][1], view_proj[1][3] - view_proj[1][1],
                                view_proj[2][3] - view_proj[2][1], view_proj[3][3] - view_proj[3][1]);

    planes[4].plane = glm::vec4(view_proj[0][3] + view_proj[0][2], view_proj[1][3] + view_proj[1][2],
                                view_proj[2][3] + view_proj[2][2], view_proj[3][3] + view_proj[3][2]);

    planes[5].plane = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

    for (int i = 0; i < 5; ++i) { // Skip far plane
        float length = glm::length(glm::vec3(planes[i].plane));
        planes[i].plane /= length;
    }

    return planes;
};


auto generate_perlin(u32 w, u32 h) -> std::vector<std::uint8_t, default_allocator<u8>>;


struct ClusterConfig {
    u32 tiles_x;
    u32 tiles_y;
    u32 tiles_z;
    u32 cluster_count;
    float z_near;
    float z_far;
    float log_z_scale;
};
auto cluster_config(u32 tiles_x, u32 tiles_y, u32 tiles_z, float z_near, float z_far) -> ClusterConfig;

struct PointLight {
    std::array<float, 4> position_radius;
    std::array<float, 4> colour_intensity;
};
auto spawn_lights_in_aabb(AABB const &aabb, std::span<PointLight> lights) -> void;
