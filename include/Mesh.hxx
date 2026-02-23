#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <future>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <tl/expected.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "BindlessHeadless.hxx"
#include "Constants.hxx"
#include "RenderContext.hxx"

#include <3PP/stb_image.h>
#include <glm/glm.hpp>

#include "AABB.hxx"
#include "Material.hxx"

auto load_mtl(const std::filesystem::path &mtl_path) -> std::unordered_map<std::string, MaterialData>;

struct TextureLoadPacket {
    enum class Type { SRGB, Linear };
    enum class Class { Albedo, Normal, Roughness, Metallic, Occlusion, Emissive };

    auto to_format() const -> VkFormat {
        switch (type) {
            case Type::Linear:
                return VK_FORMAT_R8G8B8A8_UNORM;
            case Type::SRGB:
                return VK_FORMAT_R8G8B8A8_SRGB;
        }
        return VK_FORMAT_R8G8B8A8_UNORM;
    }

    std::vector<uint8_t> rgba;
    int width{0};
    int height{0};
    Type type{Type::Linear};
    Class texture_class{Class::Albedo};
    std::string name{};
};


struct LoadedTextureCpu {
    std::string name{};
    TextureLoadPacket::Type type{};
    TextureLoadPacket::Class texture_class{};

    u32 width{0};
    u32 height{0};
    u32 levels{1};

    VkFormat vk_format{VK_FORMAT_UNDEFINED};

    std::vector<u8> data{};
    std::vector<u32> level_offset{};
    std::vector<u32> level_size{};

    auto level_span(u32 level) const -> std::span<const u8> {
        return std::span<const u8>{data.data() + level_offset[level], level_size[level]};
    }
};
auto load_texture_from_file(const std::filesystem::path &texture_path, const TextureLoadPacket::Type type,
                            const TextureLoadPacket::Class texture_class) -> TextureLoadPacket;

struct Submesh {
    u32 index_offset{0};
    u32 index_count{0};
    u32 material_id{0};
    bool alpha_tested{false};
};

struct Vertex {
    glm::vec3 position{};
    glm::vec2 uvs{};
    u32 normal{}; // packed 10_10_10_2
    u32 tangent{}; // packed 10_10_10_2 (xyz, w unused)
    u32 bitangent{}; // packed 10_10_10_2 (xyz, w unused)

    auto operator<=>(const Vertex &other) const {
        return std::tie(position.x, position.y, position.z, normal, uvs.x, uvs.y, tangent, bitangent) <=>
               std::tie(other.position.x, other.position.y, other.position.z, other.normal, other.uvs.x, other.uvs.y,
                        other.tangent, other.bitangent);
    }

    auto operator==(const Vertex &other) const -> bool = default;
};

struct PositionOnlyVertex {
    glm::vec3 pos;
};


static_assert(sizeof(Vertex) == 32);

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<u32> indices;
    std::vector<Submesh> submeshes;
};

struct DefaultTextureHandles {
    TextureHandle white{};
    TextureHandle black{};
    TextureHandle flat_normal{};
};

struct MaterialIdTable {
    std::unordered_map<std::string, u32> name_to_id;
    std::vector<std::string> id_to_name;
};

struct LoadedTextureTable {
    std::unordered_map<std::string, TextureHandle> by_stem;
};

struct LoadedObj {
    MeshData mesh;
    std::unordered_map<std::string, MaterialData> materials;
    std::vector<GPUMaterialData> gpu_materials;
    std::vector<VkDrawIndexedIndirectCommand> indirect_template;
    // GPUMaterialData
    BufferHandle material_buffer;
    // Submesh -> Material mapping
    BufferHandle material_ids_buffer;
    BufferHandle vertex_buffer;
    BufferHandle pos_uv_buffer;
    BufferHandle index_buffer;
    u32 draw_count;

    AABB mesh_aabb;
    std::vector<AABB> submesh_aabbs;
    BufferHandle aabb_buffer;
};


auto load_obj(RenderContext &ctx, const std::filesystem::path &obj_path, float scale = static_cast<float>(meters_per_unit_engine))
        -> tl::expected<LoadedObj, Error>;
