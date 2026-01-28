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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "BindlessHeadless.hxx"
#include "Constants.hxx"
#include "RenderContext.hxx"

#include <3PP/stb_image.h>
#include <glm/glm.hpp>

// -----------------------------------------------------------------------------
// Materials
// -----------------------------------------------------------------------------


struct MaterialData {
    std::string name{};
    std::string albedo_map{};
    glm::vec4 albedo_factor{1.0f};
    std::string normal_map{};
    std::string roughness_map{};
    float roughness_factor{1.0f};
    std::string metallic_map{};
    float metallic_factor{1.0f};
    std::string occlusion_map{};
    std::string emissive_map{};
    glm::vec3 emissive_factor{0.0f};
};

struct GPUMaterialData {
    u32 albedo_map{};
    glm::vec4 albedo_factor{1.0f};
    u32 normal_map{};
    u32 roughness_map{};
    float roughness_factor{1.0f};
    u32 metallic_map{};
    float metallic_factor{1.0f};
    u32 occlusion_map{};
    u32 emissive_map{};
    glm::vec3 emissive_factor{0.0f};
    u32 _pad0{}; // Ensure struct size is multiple of 16
};

auto load_mtl(const std::filesystem::path &mtl_path) -> std::unordered_map<std::string, MaterialData>;

// -----------------------------------------------------------------------------
// Textures
// -----------------------------------------------------------------------------

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

auto load_texture_from_file(const std::filesystem::path &texture_path, const TextureLoadPacket::Type type,
                            const TextureLoadPacket::Class texture_class) -> TextureLoadPacket;

// -----------------------------------------------------------------------------
// Mesh layout: one big buffer + submeshes
// -----------------------------------------------------------------------------

struct Submesh {
    u32 index_offset{0};
    u32 index_count{0};
    u32 material_id{0};
};

struct Vertex {
    glm::vec3 position;
    uint32_t normal; // packed 10_10_10_2
    uint32_t uvs; // packed 8_8_8_8

    auto operator<=>(const Vertex &other) const {
        return std::tie(position.x, position.y, position.z, normal, uvs) <=>
               std::tie(other.position.x, other.position.y, other.position.z, other.normal, other.uvs);
    }

    bool operator==(const Vertex &other) const = default;
};

static_assert(std::is_trivial_v<Vertex>);
static_assert(sizeof(Vertex) == 20);

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<u32> indices;
    std::vector<Submesh> submeshes;
};

// -----------------------------------------------------------------------------
// Defaults: global texture indices provided by you
// -----------------------------------------------------------------------------

struct DefaultTextureHandles {
    TextureHandle white{};
    TextureHandle black{};
    TextureHandle flat_normal{};
};


// -----------------------------------------------------------------------------
// Material-id mapping (string -> dense id) built during OBJ parse
// -----------------------------------------------------------------------------

struct MaterialIdTable {
    std::unordered_map<std::string, u32> name_to_id;
    std::vector<std::string> id_to_name;
};


// -----------------------------------------------------------------------------
// Resolve texture names to handles
// -----------------------------------------------------------------------------

struct LoadedTextureTable {
    std::unordered_map<std::string, TextureHandle> by_name; // key: filename (as in MTL)
};


// -----------------------------------------------------------------------------
// Loader return type: mesh + GPU material buffer + debug info
// -----------------------------------------------------------------------------

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
    BufferHandle position_vertex_buffer;
    BufferHandle index_buffer;
    u32 draw_count;
};

// -----------------------------------------------------------------------------
// OBJ loader (full integration)
// -----------------------------------------------------------------------------

auto load_obj(RenderContext &ctx, GlobalCommandContext &cmd_ctx, const std::filesystem::path &obj_path)
        -> std::optional<LoadedObj>;
