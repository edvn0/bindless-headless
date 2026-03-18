#include "AABB.hxx"
#include <format>
#include "Mesh.hxx"

namespace {
    template<typename VertexType>
    auto compute_submesh_aabbs(std::span<const VertexType> vertices, std::span<const u32> indices,
                               std::span<const Submesh> submeshes) -> Vec<AABB> {
        Vec<AABB> result;
        result.reserve(submeshes.size());

        for (auto const &submesh: submeshes) {
            AABB aabb;

            u32 const end_index = submesh.index_offset + submesh.index_count;
            for (u32 i = submesh.index_offset; i < end_index; ++i) {
                u32 const vertex_index = indices[i];
                aabb.expand(vertices[vertex_index].position);
            }

            result.push_back(aabb);
        }

        return result;
    }
} // namespace
auto merge_aabbs(std::span<const AABB> aabbs) -> AABB {
    AABB result;

    for (auto const &aabb: aabbs) {
        result.expand(aabb);
    }

    return result;
}


auto create_aabb_device_buffer(VmaAllocator &allocator, std::span<const AABB> aabbs, std::string const &name)
        -> tl::expected<Buffer, Error> {

    std::vector<PackedAABB> packed_aabbs;
    packed_aabbs.reserve(aabbs.size());

    for (auto const &aabb: aabbs) {
        packed_aabbs.emplace_back(aabb);
    }

    return Buffer::from_slice<PackedAABB>(allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, std::span(packed_aabbs), name);
}

auto update_aabb_device_buffer(VmaAllocator &allocator, Buffer &buffer, std::span<const AABB> aabbs) -> void {

    std::vector<PackedAABB> packed_aabbs;
    packed_aabbs.reserve(aabbs.size());

    for (auto const &aabb: aabbs) {
        packed_aabbs.emplace_back(aabb);
    }

    buffer.write_slice(allocator, std::span(packed_aabbs), 0);
}


auto create_mesh_aabb_data(VmaAllocator &allocator, MeshData const &mesh, std::string const &name)
        -> tl::expected<MeshAABBData, Error> {

    auto submesh_aabbs =
            compute_submesh_aabbs(std::span(mesh.vertices), std::span(mesh.indices), std::span(mesh.submeshes));

    const auto mesh_aabb = merge_aabbs(std::span(submesh_aabbs));

    // Create device buffer
    auto buffer_result =
            create_aabb_device_buffer(allocator, std::span(submesh_aabbs), std::format("aabb_buffer_{}", name));

    if (!buffer_result) {
        return tl::make_unexpected(buffer_result.error());
    }

    return MeshAABBData{
            .mesh_aabb = mesh_aabb,
            .submesh_aabbs = std::move(submesh_aabbs),
            .device_buffer = std::move(buffer_result.value()),
    };
}
