#pragma once

#include <glm/glm.hpp>
#include <limits>
#include <vector>


#include "Buffer.hxx"
#include "Forward.hxx"
#include "Types.hxx"


struct AABB {
    glm::vec3 min;
    glm::vec3 max;

    constexpr AABB() : min(std::numeric_limits<float>::max()), max(std::numeric_limits<float>::lowest()) {}

    constexpr AABB(glm::vec3 const &min_point, glm::vec3 const &max_point) : min(min_point), max(max_point) {}

    [[nodiscard]] constexpr auto center() const -> glm::vec3 { return (min + max) * 0.5f; }

    [[nodiscard]] constexpr auto extent() const -> glm::vec3 { return max - min; }

    [[nodiscard]] constexpr auto half_extent() const -> glm::vec3 { return (max - min) * 0.5f; }

    [[nodiscard]] constexpr auto is_valid() const -> bool { return min.x <= max.x && min.y <= max.y && min.z <= max.z; }

    [[nodiscard]] constexpr auto surface_area() const -> float {
        glm::vec3 ext = extent();
        return 2.0f * (ext.x * ext.y + ext.y * ext.z + ext.z * ext.x);
    }

    [[nodiscard]] constexpr auto volume() const -> float {
        glm::vec3 ext = extent();
        return ext.x * ext.y * ext.z;
    }

    constexpr auto expand(glm::vec3 const &point) -> void {
        min = glm::min(min, point);
        max = glm::max(max, point);
    }

    constexpr auto expand(const std::array<float, 3> &point) -> void {
        expand(glm::vec3(point[0], point[1], point[2]));
    }

    constexpr auto expand(AABB const &other) -> void {
        min = glm::min(min, other.min);
        max = glm::max(max, other.max);
    }

    [[nodiscard]] constexpr auto contains(glm::vec3 const &point) const -> bool {
        return point.x >= min.x && point.x <= max.x && point.y >= min.y && point.y <= max.y && point.z >= min.z &&
               point.z <= max.z;
    }

    [[nodiscard]] constexpr auto intersects(AABB const &other) const -> bool {
        return (min.x <= other.max.x && max.x >= other.min.x) && (min.y <= other.max.y && max.y >= other.min.y) &&
               (min.z <= other.max.z && max.z >= other.min.z);
    }

    [[nodiscard]] constexpr auto transform(glm::mat4 const &matrix) const -> AABB {
        AABB result;

        // Transform all 8 corners of the AABB
        std::array<glm::vec3, 8> corners = {
                glm::vec3{min.x, min.y, min.z}, {min.x, min.y, max.z}, {min.x, max.y, min.z}, {min.x, max.y, max.z},
                {max.x, min.y, min.z},          {max.x, min.y, max.z}, {max.x, max.y, min.z}, {max.x, max.y, max.z},
        };

        for (auto const &corner: corners) {
            glm::vec4 transformed = matrix * glm::vec4(corner, 1.0f);
            result.expand(glm::vec3(transformed) / transformed.w);
        }

        return result;
    }

    [[nodiscard]] constexpr auto scaled(std::floating_point auto scale) const -> AABB {
        return AABB{min * scale, max * scale};
    }
};

// GPU-compatible packed AABB structure
struct PackedAABB {
    glm::vec3 min;
    u32 _pad0;
    glm::vec3 max;
    u32 _pad1;

    PackedAABB() = default;

    explicit PackedAABB(AABB const &aabb) : min(aabb.min), _pad0(0), max(aabb.max), _pad1(0) {}

    [[nodiscard]] constexpr auto to_aabb() const -> AABB { return AABB{min, max}; }
};

static_assert(sizeof(PackedAABB) == 32, "PackedAABB must be 32 bytes for GPU alignment");

// Compute AABB from vertex positions
template<typename VertexType>
auto compute_aabb_from_vertices(std::span<const VertexType> vertices) -> AABB {
    AABB result;

    for (auto const &vertex: vertices) {
        result.expand(vertex.position);
    }

    return result;
}

auto merge_aabbs(std::span<const AABB> aabbs) -> AABB;
auto compute_instance_aabb(AABB const &local_aabb, glm::mat4 const &transform) -> AABB;
struct MeshAABBData {
    AABB mesh_aabb;
    std::vector<AABB> submesh_aabbs;
    Buffer device_buffer;
};
auto create_mesh_aabb_data(VmaAllocator &allocator, MeshData const &mesh, std::string const &name)
        -> tl::expected<MeshAABBData, Error>;
