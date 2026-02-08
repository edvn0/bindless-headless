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


glm::mat4 PerspectiveRH_ReverseZ_Inf(float fovYRadians, float aspect, float zNear);

constexpr auto extract_frustum_planes = [](const glm::mat4 &inv_proj) -> std::array<FrustumPlane, 6> {
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
auto cluster_config(u32 tiles_x, u32 tiles_y, u32 tiles_z, float z_near, float z_far)->ClusterConfig;

struct PointLight {
    std::array<float, 4> position_radius;
    std::array<float, 4> colour_intensity;
};
auto spawn_lights_in_aabb(AABB const &aabb, std::span<PointLight> lights) -> void;