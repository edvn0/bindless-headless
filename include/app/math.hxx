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

auto current_extent(GLFWwindow *win) -> VkExtent2D;

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


constexpr auto extract_frustum_planes = [](const glm::mat4 &proj) -> std::array<FrustumPlane, 6> {
    std::array<FrustumPlane, 6> planes{};

    // Each plane is row3 +/- rowN
    // Row 0: proj[0][0], proj[1][0], proj[2][0], proj[3][0]
    // Row 3: proj[0][3], proj[1][3], proj[2][3], proj[3][3]

    glm::vec4 row0 = {proj[0][0], proj[1][0], proj[2][0], proj[3][0]};
    glm::vec4 row1 = {proj[0][1], proj[1][1], proj[2][1], proj[3][1]};
    glm::vec4 row2 = {proj[0][2], proj[1][2], proj[2][2], proj[3][2]};
    glm::vec4 row3 = {proj[0][3], proj[1][3], proj[2][3], proj[3][3]};

    planes[0].plane = row3 + row0; // left
    planes[1].plane = row3 - row0; // right
    planes[2].plane = row3 + row1; // bottom
    planes[3].plane = row3 - row1; // top
    planes[4].plane = row3 + row2; // near  (reverse-Z: this becomes far in NDC)
    planes[5].plane = row3 - row2; // far   (reverse-Z: this becomes near in NDC)


    for (auto &p: planes) {
        float len = glm::length(glm::vec3(p.plane));
        p.plane /= len;
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
    std::array<float, 4> position_radius{};
    std::array<float, 4> colour_intensity{};
};
auto spawn_lights_in_aabb(AABB const &aabb, std::span<PointLight> lights) -> void;
