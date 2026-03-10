#include "app/math.hxx"
#include <random>

#include "AABB.hxx"

#include "3PP/PerlinNoise.hpp"

auto generate_perlin(u32 w, u32 h) -> std::vector<std::uint8_t, default_allocator<u8>> {
    std::vector<std::uint8_t, default_allocator<u8>> data;
    data.resize(w * h);
    const auto seed = static_cast<u32>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const siv::PerlinNoise pn{seed};

    auto z_offset = 0.0;
    for (u32 y = 0; y < h; ++y) {
        const auto row_z = z_offset + static_cast<double>(y) * 0.01;
        for (u32 x = 0; x < w; ++x) {
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


auto spawn_lights_in_aabb(AABB const &aabb, std::span<PointLight> lights) -> void {
    thread_local auto rng = std::default_random_engine{};

    auto x_distrib = std::uniform_real_distribution{aabb.min.x, aabb.max.x};
    auto y_distrib = std::uniform_real_distribution{aabb.min.y, aabb.max.y};
    auto z_distrib = std::uniform_real_distribution{aabb.min.z, aabb.max.z};
    auto radius_distrib = std::uniform_real_distribution{2.f, 6.f};
    auto intensity_distrib = std::uniform_real_distribution{0.5F, 5.0F};

    auto color_distribution = std::uniform_real_distribution{0.0F, 1.0F};

    for (auto &[position_radius, colour_intensity]: lights) {
        auto const intensity = intensity_distrib(rng);
        auto const radius = radius_distrib(rng);

        position_radius = {x_distrib(rng), y_distrib(rng), z_distrib(rng), radius};

        colour_intensity = {color_distribution(rng), color_distribution(rng), color_distribution(rng), intensity};
    }
};
auto PerspectiveRH_ReverseZ_Inf(float fovYRadians, float aspect, float zNear) -> glm::mat4 {
    const float f = 1.0f / glm::tan(fovYRadians * 0.5f);

    glm::mat4 m{0.0f};

    m[0][0] = f / aspect;
    m[1][1] = f;
    m[2][3] = -1.0f;
    m[3][2] = zNear;

    m[2][2] = 0.0f;

    return m;
}

auto OrthoRH_ReverseZ(float left, float right, float bottom, float top, float near_plane, float far_plane)
        -> glm::mat4 {
    glm::mat4 result(1.0f);

    result[0][0] = 2.0f / (right - left);
    result[1][1] = 2.0f / (top - bottom);
    result[2][2] = 1.0f / (near_plane - far_plane); // Reversed for reverse-Z
    result[3][0] = -(right + left) / (right - left);
    result[3][1] = -(top + bottom) / (top - bottom);
    result[3][2] = near_plane / (near_plane - far_plane); // Reversed for reverse-Z

    return result;
}

auto cluster_config(u32 tiles_x, u32 tiles_y, u32 tiles_z, float near_plane, float far_plane) -> ClusterConfig {
    u32 cluster_count = tiles_x * tiles_y * tiles_z;
    float log_z_scale = static_cast<float>(tiles_z) / std::log2f(far_plane / near_plane);

    return ClusterConfig{
            .tiles_x = tiles_x,
            .tiles_y = tiles_y,
            .tiles_z = tiles_z,
            .cluster_count = cluster_count,
            .z_near = near_plane,
            .z_far = far_plane,
            .log_z_scale = log_z_scale,
    };
}

auto current_extent(GLFWwindow *win) -> VkExtent2D {
    int fbw{0};
    int fbh{0};
    glfwGetFramebufferSize(win, &fbw, &fbh);
    return VkExtent2D{.width = static_cast<u32>(std::max(fbw, 0)), .height = static_cast<u32>(std::max(fbh, 0))};
};
