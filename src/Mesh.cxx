#include "Mesh.hxx"
#include "CompilerGlue.hxx"
#include "Error.hxx"
#include "Logger.hxx"
#include "Profiler.hxx"
#include "RenderContext.hxx"
#include "SceneLoader.hxx" // FileHeader, Submesh (file), GPUMaterial, Texture, BlobRange, etc.
#include "Types.hxx"


#include <bzlib.h>
#include <ktx.h>
#include <ktxvulkan.h>
#include <volk.h>

#include <bit>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <span>
#include <string_view>
#include <vector>

#include <tl/expected.hpp>

#include <glm/gtc/packing.hpp>
#include <glm/packing.hpp>
#include <ktx.h>
#include <ktxvulkan.h>

#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include <tiny_obj_loader.h>


namespace {

    constexpr auto is_normal_mode = [](auto texture) -> bool {
        char *value{};
        u32 length{};

        if (KTX_SUCCESS ==
            ktxHashList_FindValue(&texture->kvDataHead, "KTXwriterScParams", &length, (void **) &value)) {
            std::string params(value, length);

            if (params.find("--normal-mode") != std::string::npos) {
                return true;
            }
        }
        return false;
    };

    static auto pick_vk_format(TextureLoadPacket::Type type) -> VkFormat {
        switch (type) {
            case TextureLoadPacket::Type::SRGB:
                return VK_FORMAT_R8G8B8A8_SRGB;
            case TextureLoadPacket::Type::Linear:
            default:
                return VK_FORMAT_R8G8B8A8_UNORM;
        }
    }

    auto load_with_stb(std::filesystem::path const &texture_path, TextureLoadPacket::Type type,
                       TextureLoadPacket::Class texture_class) -> LoadedTextureCpu {
        LoadedTextureCpu out{};
        out.name = texture_path.filename().string();
        out.type = type;
        out.texture_class = texture_class;
        out.vk_format = pick_vk_format(type);

        int width = 0, height = 0, channels = 0;
        // stbi_set_flip_vertically_on_load(true);

        unsigned char *pixels = stbi_load(texture_path.string().c_str(), &width, &height, &channels, 4);
        if (!pixels) {
            out.width = 1;
            out.height = 1;
            out.levels = 1;
            out.data = {255, 0, 255, 255};
            out.level_offset = {0};
            out.level_size = {4};
            return out;
        }

        out.width = static_cast<u32>(width);
        out.height = static_cast<u32>(height);
        out.levels = 1;

        u32 const size = out.width * out.height * 4;
        out.data.assign(pixels, pixels + size);
        out.level_offset = {0};
        out.level_size = {size};

        stbi_image_free(pixels);
        return out;
    }


    auto load_ktx2_cpu_bc7(std::filesystem::path const &texture_path, TextureLoadPacket::Type type,
                           TextureLoadPacket::Class texture_class) -> LoadedTextureCpu {
        LoadedTextureCpu out{};
        out.name = texture_path.filename().string();
        out.type = type;
        out.texture_class = texture_class;

        ktxTexture2 *ktx2 = nullptr;
        KTX_error_code res =
                ktxTexture_CreateFromNamedFile(texture_path.string().c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT,
                                               reinterpret_cast<ktxTexture **>(&ktx2));

        if (res != KTX_SUCCESS || !ktx2) {
            out.width = 1;
            out.height = 1;
            out.levels = 1;
            out.vk_format = pick_vk_format(type);
            out.data = {255, 0, 255, 255};
            out.level_offset = {0};
            out.level_size = {4};
            return out;
        }

        auto is_normal_map = is_normal_mode(ktx2);

        if (ktxTexture2_NeedsTranscoding(ktx2)) {
            ktx_transcode_fmt_e target_format = KTX_TTF_BC7_RGBA;


            res = ktxTexture2_TranscodeBasis(ktx2, target_format, 0);
            if (res != KTX_SUCCESS) {
                ktxTexture2_Destroy(ktx2);
                out.width = 1;
                out.height = 1;
                out.levels = 1;
                out.vk_format = pick_vk_format(type);
                out.data = {255, 0, 255, 255};
                out.level_offset = {0};
                out.level_size = {4};
                return out;
            }

            if (is_normal_map) {
                out.vk_format = VK_FORMAT_BC7_UNORM_BLOCK;
            } else {
                out.vk_format =
                        (type == TextureLoadPacket::Type::SRGB) ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
            }
        } else {
            out.vk_format = static_cast<VkFormat>(ktx2->vkFormat);
        }

        out.width = static_cast<u32>(ktx2->baseWidth);
        out.height = static_cast<u32>(ktx2->baseHeight);
        out.levels = static_cast<u32>(ktx2->numLevels);

        out.level_offset.resize(out.levels);
        out.level_size.resize(out.levels);

        u32 total = 0;
        for (u32 level = 0; level < out.levels; ++level) {
            ktx_size_t off = 0;
            res = ktxTexture_GetImageOffset(reinterpret_cast<ktxTexture *>(ktx2), level, 0, 0, &off);
            if (res != KTX_SUCCESS) {
                ktxTexture_Destroy(reinterpret_cast<ktxTexture *>(ktx2));
                out = {};
                out.name = texture_path.filename().string();
                out.type = type;
                out.texture_class = texture_class;
                out.width = 1;
                out.height = 1;
                out.levels = 1;
                out.vk_format = pick_vk_format(type);
                out.data = {255, 0, 255, 255};
                out.level_offset = {0};
                out.level_size = {4};
                return out;
            }

            auto level_size = ktxTexture_GetImageSize(reinterpret_cast<ktxTexture *>(ktx2), level);
            if (level_size == 0) {
                ktxTexture_Destroy(reinterpret_cast<ktxTexture *>(ktx2));
                out = {};
                out.name = texture_path.filename().string();
                out.type = type;
                out.texture_class = texture_class;
                out.width = 1;
                out.height = 1;
                out.levels = 1;
                out.vk_format = pick_vk_format(type);
                out.data = {255, 0, 255, 255};
                out.level_offset = {0};
                out.level_size = {4};
                return out;
            }

            out.level_offset[level] = total;
            out.level_size[level] = static_cast<u32>(level_size);
            total += static_cast<u32>(level_size);
        }

        out.data.resize(total);

        u8 const *base = reinterpret_cast<u8 const *>(ktxTexture_GetData(reinterpret_cast<ktxTexture *>(ktx2)));

        for (u32 level = 0; level < out.levels; ++level) {
            ktx_size_t off = 0;
            (void) ktxTexture_GetImageOffset(reinterpret_cast<ktxTexture *>(ktx2), level, 0, 0, &off);

            std::memcpy(out.data.data() + out.level_offset[level], base + off, out.level_size[level]);
        }

        ktxTexture_Destroy(reinterpret_cast<ktxTexture *>(ktx2));
        return out;
    }

    auto load_texture_unified(std::filesystem::path const &texture_path, TextureLoadPacket::Type type,
                              TextureLoadPacket::Class texture_class) -> LoadedTextureCpu {
        auto ext = texture_path.extension().string();
        std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });

        if (ext == ".ktx2") {
            return load_ktx2_cpu_bc7(texture_path, type, texture_class);
        }

        return load_with_stb(texture_path, type, texture_class);
    }

    auto unpack_normal(u32 packed) -> glm::vec3 {
        const glm::vec4 n4 = glm::unpackSnorm3x10_1x2(packed);
        return glm::vec3{n4.x, n4.y, n4.z};
    }

    auto pack_dir(glm::vec3 v) -> u32 { return glm::packSnorm3x10_1x2(glm::vec4{v, 0.0f}); }

    auto safe_normalize(glm::vec3 v, float eps = 1e-20f) -> glm::vec3 {
        const float len2 = glm::dot(v, v);
        if (len2 <= eps)
            return glm::vec3{0.0f};
        return v * glm::inversesqrt(len2);
    }

    auto build_any_orthonormal_tangent(glm::vec3 n) -> glm::vec3 {
        const glm::vec3 a = (std::abs(n.z) < 0.999f) ? glm::vec3{0.0f, 0.0f, 1.0f} : glm::vec3{0.0f, 1.0f, 0.0f};
        return safe_normalize(glm::cross(a, n));
    }

    auto compute_tangent_basis(MeshData &mesh) -> void {
        if (mesh.vertices.empty() || mesh.indices.size() < 3)
            return;

        const bool has_any_tangent = std::ranges::any_of(
                mesh.vertices, [](const Vertex &v) { return v.tangent != 0u || v.bitangent != 0u; });
        if (has_any_tangent)
            return;

        std::vector<glm::vec3> tan_acc(mesh.vertices.size(), glm::vec3{0.0f});
        std::vector<glm::vec3> bitan_acc(mesh.vertices.size(), glm::vec3{0.0f});

        constexpr float eps = 1e-12f;

        // Accumulate per-triangle tangents/bitangents.
        for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
            const u32 i0 = mesh.indices[i + 0];
            const u32 i1 = mesh.indices[i + 1];
            const u32 i2 = mesh.indices[i + 2];

            const Vertex &v0 = mesh.vertices[i0];
            const Vertex &v1 = mesh.vertices[i1];
            const Vertex &v2 = mesh.vertices[i2];

            const glm::vec3 p0 = v0.position;
            const glm::vec3 p1 = v1.position;
            const glm::vec3 p2 = v2.position;

            const auto uv0 = glm::unpackHalf2x16(v0.uvs);
            const auto uv1 = glm::unpackHalf2x16(v1.uvs);
            const auto uv2 = glm::unpackHalf2x16(v2.uvs);

            const glm::vec3 e1 = p1 - p0;
            const glm::vec3 e2 = p2 - p0;

            const glm::vec2 duv1 = uv1 - uv0;
            const glm::vec2 duv2 = uv2 - uv0;

            const float denom = duv1.x * duv2.y - duv1.y * duv2.x;
            if (std::abs(denom) <= eps) {
                continue;
            }

            const float r = 1.0f / denom;
            glm::vec3 t = (e1 * duv2.y - e2 * duv1.y) * r;
            glm::vec3 b = (e2 * duv1.x - e1 * duv2.x) * r;

            const glm::vec3 c = glm::cross(e1, e2);
            const float area_w = glm::length(c);
            t *= area_w;
            b *= area_w;

            tan_acc[i0] += t;
            tan_acc[i1] += t;
            tan_acc[i2] += t;
            bitan_acc[i0] += b;
            bitan_acc[i1] += b;
            bitan_acc[i2] += b;
        }

        for (size_t vi = 0; vi < mesh.vertices.size(); ++vi) {
            Vertex &v = mesh.vertices[vi];

            glm::vec3 n = safe_normalize(unpack_normal(v.normal));
            if (glm::dot(n, n) <= 0.0f) {
                n = glm::vec3{0.0f, 1.0f, 0.0f};
            }

            glm::vec3 t = tan_acc[vi];
            t = safe_normalize(t);

            if (glm::dot(t, t) <= 0.0f) {
                t = build_any_orthonormal_tangent(n);
            } else {
                t = safe_normalize(t - n * glm::dot(n, t));
                if (glm::dot(t, t) <= 0.0f)
                    t = build_any_orthonormal_tangent(n);
            }

            const glm::vec3 b_acc = bitan_acc[vi];
            glm::vec3 b = glm::cross(n, t);
            const float handed = (glm::dot(b, b_acc) < 0.0f) ? -1.0f : 1.0f;
            b *= handed;

            v.tangent = pack_dir(t);
            v.bitangent = pack_dir(safe_normalize(b));
        }
    }

    auto get_default_texture_handles(const RenderContext &ctx) -> DefaultTextureHandles {
        return {
                .white = ctx.textures.get_handle(white_texture_index),
                .black = ctx.textures.get_handle(black_texture_index),
                .flat_normal = ctx.textures.get_handle(normal_texture_index),
        };
    }

    auto get_or_create_material_id(MaterialIdTable &table, const std::string &name) -> u32 {
        auto it = table.name_to_id.find(name);
        if (it != table.name_to_id.end())
            return it->second;

        u32 id = static_cast<u32>(table.id_to_name.size());
        table.name_to_id.emplace(name, id);
        table.id_to_name.emplace_back(name);
        return id;
    }

    auto build_loaded_texture_table(std::span<const LoadedTextureCpu> textures, std::span<const TextureHandle> handles)
            -> LoadedTextureTable {
        LoadedTextureTable out{};
        out.by_stem.reserve(textures.size());
        for (size_t i = 0; i < textures.size(); ++i) {
            // Convert "brick.ktx2" or "brick.jpg" -> "brick"
            std::string stem = std::filesystem::path(textures[i].name).stem().string();
            out.by_stem.emplace(stem, handles[i]);
        }
        return out;
    }

    auto resolve_texture(const LoadedTextureTable &loaded, const std::string &name, TextureHandle fallback) -> u32 {
        if (name.empty())
            return fallback.index();

        std::string stem = std::filesystem::path(name).stem().string();

        auto it = loaded.by_stem.find(stem);
        if (it != loaded.by_stem.end()) {
            return it->second.index();
        }

        // Optional: Log a warning if a texture was expected but not found
        warn("Texture stem not found: {}", stem);
        return fallback.index();
    }

    auto resolve_texture_path(const std::filesystem::path &base_path, const std::string &tex_name)
            -> std::filesystem::path {
        std::filesystem::path original_path(tex_name);
        std::filesystem::path ktx2_name = original_path.stem().replace_extension(".ktx2");

        if (std::filesystem::exists(base_path / ktx2_name)) {
            return base_path / ktx2_name;
        }

        // 2. Check for .ktx2 in the 'textures/' subdirectory
        if (std::filesystem::exists(base_path / "textures" / ktx2_name)) {
            return base_path / "textures" / ktx2_name;
        }

        if (std::filesystem::exists(base_path / "textures" / "ktx_compressed" / ktx2_name)) {
            return base_path / "textures" / "ktx_compressed" / ktx2_name;
        }

        if (std::filesystem::exists(base_path / "textures" / "ktx-compressed" / ktx2_name)) {
            return base_path / "textures" / "ktx-compressed" / ktx2_name;
        }

        if (std::filesystem::exists(base_path / "textures" / "ktx" / ktx2_name)) {
            return base_path / "textures" / "ktx" / ktx2_name;
        }

        if (std::filesystem::exists(base_path / "textures" / "ktx2" / ktx2_name)) {
            return base_path / "textures" / "ktx2" / ktx2_name;
        }

        std::filesystem::path primary = base_path / tex_name;
        if (std::filesystem::exists(primary)) {
            return primary;
        }

        std::filesystem::path secondary = base_path / "textures" / tex_name;
        if (std::filesystem::exists(secondary)) {
            return secondary;
        }

        return primary;
    }

    auto to_gpu_material(const MaterialData &m, const LoadedTextureTable &loaded, const DefaultTextureHandles &defs)
            -> GPUMaterialData {
        GPUMaterialData out{};
        out.albedo_map = resolve_texture(loaded, m.albedo_map, defs.white);
        out.albedo_factor = m.albedo_factor;
        out.set_albedo_map(out.albedo_map != defs.white.index());

        out.normal_map = resolve_texture(loaded, m.normal_map, defs.flat_normal);
        out.set_normal_map(out.normal_map != defs.flat_normal.index());

        out.roughness_map = resolve_texture(loaded, m.roughness_map, defs.white);
        out.roughness_factor = m.roughness_factor;
        out.set_roughness_map(out.roughness_map != defs.white.index());

        out.metallic_map = resolve_texture(loaded, m.metallic_map, defs.black);
        out.metallic_factor = m.metallic_factor;
        out.set_metallic_map(out.metallic_map != defs.black.index());

        out.occlusion_map = resolve_texture(loaded, m.occlusion_map, defs.white);
        out.set_occlusion_map(out.occlusion_map != defs.white.index());

        out.emissive_map = resolve_texture(loaded, m.emissive_map, defs.black);
        out.emissive_factor = m.emissive_factor;
        out.set_emissive_map(out.emissive_map != defs.black.index());

        out.set_is_alpha_tested(m.is_alpha_tested);

        return out;
    }

    auto to_material_data(const tinyobj::material_t &m) -> MaterialData {
        MaterialData out{};
        out.name = m.name;

        out.albedo_factor = glm::vec4{m.diffuse[0], m.diffuse[1], m.diffuse[2], 1.0f};
        out.albedo_map = m.diffuse_texname;

        out.normal_map = first_non_empty(m.bump_texname, m.normal_texname, m.displacement_texname);

        out.roughness_factor = m.roughness;
        out.metallic_factor = m.metallic;
        out.roughness_map = m.roughness_texname;
        out.metallic_map = m.metallic_texname;

        out.occlusion_map = m.ambient_texname;

        out.emissive_factor = glm::vec3{m.emission[0], m.emission[1], m.emission[2]};
        out.emissive_map = m.emissive_texname;

        out.is_alpha_tested = (m.dissolve > 0.0f) || (!m.alpha_texname.empty());

        if (out.albedo_factor == glm::vec4(0.0f))
            out.albedo_factor = glm::vec4{1.0f};

        static constexpr auto default_transparency_if_alpha_tested = 0.1F;
        out.albedo_factor.a = out.is_alpha_tested ? default_transparency_if_alpha_tested : 1.0F;
        return out;
    }

    struct VertexHash {
        auto operator()(const Vertex &v) const noexcept -> size_t {
            size_t h = 0;
            h ^= std::hash<float>()(v.position[0]) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<float>()(v.position[1]) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<float>()(v.position[2]) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<u32>()(v.uvs) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<u32>()(v.normal) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<u32>()(v.tangent) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<u32>()(v.bitangent) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };

    auto pack_vertex_from_indices(const tinyobj::attrib_t &attrib, const tinyobj::index_t &idx) -> Vertex {
        Vertex v{};

        if (idx.vertex_index >= 0) {
            const int base = 3 * idx.vertex_index;
            v.position = {
                    attrib.vertices[base + 0],
                    attrib.vertices[base + 1],
                    attrib.vertices[base + 2],
            };
        } else {
            v.position = glm::vec3(0.0f);
        }

        glm::vec3 n{0.0f, 1.0f, 0.0f};
        if (idx.normal_index >= 0 && !attrib.normals.empty()) {
            const int base = 3 * idx.normal_index;
            n = glm::vec3{
                    attrib.normals[base + 0],
                    attrib.normals[base + 1],
                    attrib.normals[base + 2],
            };
        }
        v.normal = glm::packSnorm3x10_1x2(glm::vec4{n, 0.0f});

        glm::vec2 uv0{0.0f};
        if (idx.texcoord_index >= 0 && !attrib.texcoords.empty()) {
            const int base = 2 * idx.texcoord_index;
            uv0 = glm::vec2{
                    attrib.texcoords[base + 0],
                    attrib.texcoords[base + 1],
            };
        }
        uv0.y = 1.0f - uv0.y;
        v.uvs = glm::packHalf2x16(uv0);
        return v;
    }
} // namespace

auto load_texture_from_file(const std::filesystem::path &texture_path, const TextureLoadPacket::Type type,
                            const TextureLoadPacket::Class texture_class) -> TextureLoadPacket {
    TextureLoadPacket packet{.rgba = {},
                             .width = 0,
                             .height = 0,
                             .type = type,
                             .texture_class = texture_class,
                             .name = texture_path.filename().string()};

    int width, height, channels;
    unsigned char *data = stbi_load(texture_path.string().c_str(), &width, &height, &channels, 4);
    if (data) {
        packet.width = width;
        packet.height = height;
        packet.rgba.assign(data, data + (width * height * 4));
        stbi_image_free(data);
    } else {
        packet.width = 1;
        packet.height = 1;
        packet.rgba = {255, 0, 255, 255};
    }

    return packet;
}

auto load_static_mesh(RenderContext &ctx, const std::filesystem::path &obj_path, float scale)
        -> tl::expected<StaticMesh, Error> {

    tinyobj::ObjReaderConfig cfg{};
    cfg.mtl_search_path = obj_path.parent_path().string();
    cfg.triangulate = true;
    cfg.triangulation_method = "earcut";

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(obj_path.string(), cfg)) {
        auto error = reader.Error();
        info("Failed to load OBJ file: {}", error);
        return tl::make_unexpected(Error::make_error(Error::Type::MeshLoadError, error));
    }

    const auto &attrib = reader.GetAttrib();
    const auto &shapes = reader.GetShapes();
    const auto &tiny_mats = reader.GetMaterials();

    MeshData mesh{};
    std::unordered_map<Vertex, u32, VertexHash> vertex_map;

    std::unordered_map<std::string, MaterialData> materials;
    materials.reserve(tiny_mats.size() + 1);

    MaterialIdTable material_ids{};

    const std::string default_name = "default";
    (void) get_or_create_material_id(material_ids, default_name);

    for (const auto &m: tiny_mats) {
        MaterialData md = to_material_data(m);
        if (md.name.empty())
            md.name = default_name;

        (void) get_or_create_material_id(material_ids, md.name);
        materials.emplace(md.name, std::move(md));
    }

    if (!materials.contains(default_name)) {
        materials[default_name] = MaterialData{
                .name = default_name,
                .albedo_factor = glm::vec4{1.0f},
                .roughness_factor = 1.0f,
                .metallic_factor = 0.0f,
                .emissive_factor = glm::vec3{0.0f},
        };
    }

    std::unordered_map<u32, std::vector<std::array<tinyobj::index_t, 3>>> material_groups;

    for (const auto &shape: shapes) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            const int fv = shape.mesh.num_face_vertices[f];
            const int mat_id = (f < shape.mesh.material_ids.size()) ? shape.mesh.material_ids[f] : -1;

            // Resolve material name/ID
            const std::string &mat_name = (mat_id >= 0 && static_cast<size_t>(mat_id) < tiny_mats.size())
                                                  ? tiny_mats[mat_id].name
                                                  : default_name;
            const u32 mat_u32 = get_or_create_material_id(material_ids, mat_name);

            // Store face indices (assuming triangulation is on)
            if (fv == 3) {
                material_groups[mat_u32].push_back({shape.mesh.indices[index_offset + 0],
                                                    shape.mesh.indices[index_offset + 1],
                                                    shape.mesh.indices[index_offset + 2]});
            }
            index_offset += fv;
        }
    }

    for (u32 m_id = 0; m_id < material_ids.id_to_name.size(); ++m_id) {
        if (!material_groups.contains(m_id))
            continue;

        const auto &faces = material_groups[m_id];
        u32 submesh_start_index = static_cast<u32>(mesh.indices.size());

        for (const auto &face: faces) {
            for (const auto &idx: face) {
                Vertex v = pack_vertex_from_indices(attrib, idx);
                v.position *= scale;

                auto it = vertex_map.find(v);
                if (it != vertex_map.end()) {
                    mesh.indices.push_back(it->second);
                } else {
                    u32 new_index = static_cast<u32>(mesh.vertices.size());
                    mesh.vertices.push_back(v);
                    vertex_map.emplace(v, new_index);
                    mesh.indices.push_back(new_index);
                }
            }
        }

        const std::string &mat_name = material_ids.id_to_name[m_id];
        bool is_alpha = materials[mat_name].is_alpha_tested;

        mesh.submeshes.push_back(Submesh{
                .index_offset = submesh_start_index,
                .index_count = static_cast<u32>(mesh.indices.size()) - submesh_start_index,
                .material_id = m_id,
                .alpha_tested = is_alpha,
        });
    }

    compute_tangent_basis(mesh);

    std::vector<std::future<LoadedTextureCpu>> load_futures;
    std::unordered_set<std::string> unique_texture_names;
    const std::filesystem::path base_path = obj_path.parent_path();

#define LOAD_MAP(mat, field_name, t, clazz)                                                                            \
    do {                                                                                                               \
        if (!(mat).field_name.empty() && !unique_texture_names.contains((mat).field_name)) {                           \
            load_futures.emplace_back(                                                                                 \
                    std::async(std::launch::async, [base_path, tex_name = (mat).field_name]() -> LoadedTextureCpu {    \
                        auto resolved = resolve_texture_path(base_path, tex_name);                                     \
                        return load_texture_unified(resolved, TextureLoadPacket::Type::t,                              \
                                                    TextureLoadPacket::Class::clazz);                                  \
                    }));                                                                                               \
            unique_texture_names.emplace((mat).field_name);                                                            \
        }                                                                                                              \
    } while (false)

    for (const auto &[_, m]: materials) {
        LOAD_MAP(m, albedo_map, SRGB, Albedo);
        LOAD_MAP(m, normal_map, Linear, Normal);
        LOAD_MAP(m, roughness_map, Linear, Roughness);
        LOAD_MAP(m, metallic_map, Linear, Metallic);
        LOAD_MAP(m, occlusion_map, Linear, Occlusion);
        LOAD_MAP(m, emissive_map, Linear, Emissive);
    }

#undef LOAD_MAP

    std::vector<LoadedTextureCpu> textures;
    textures.reserve(load_futures.size());
    for (auto &f: load_futures) {
        textures.emplace_back(f.get());
    }

    std::vector<TextureHandle> handles;
    handles.reserve(textures.size());

    for (auto const &tex: textures) {
        auto img =
                create_texture_image_v2(ctx.allocator, *ctx.command_ctx, tex.width, tex.height, tex.vk_format,
                                        std::span<const u8>{tex.data.data(), tex.data.size()},
                                        std::span<const u32>{tex.level_offset.data(), tex.level_offset.size()},
                                        std::span<const u32>{tex.level_size.data(), tex.level_size.size()}, tex.name);

        handles.emplace_back(ctx.textures.create(std::move(img)));
    }

    DefaultTextureHandles defs = get_default_texture_handles(ctx);
    LoadedTextureTable loaded = build_loaded_texture_table(std::span{textures}, handles);

    std::vector<GPUMaterialData> gpu_materials;
    gpu_materials.reserve(material_ids.id_to_name.size());

    for (const std::string &mat_name: material_ids.id_to_name) {
        auto it = materials.find(mat_name);
        if (it == materials.end()) {
            MaterialData fallback{};
            fallback.name = mat_name;
            fallback.albedo_factor = glm::vec4{1.0f};
            fallback.roughness_factor = 1.0f;
            fallback.metallic_factor = 0.0f;
            gpu_materials.push_back(to_gpu_material(fallback, loaded, defs));
        } else {
            gpu_materials.push_back(to_gpu_material(it->second, loaded, defs));
        }
    }

    const auto &vb_copy = mesh.vertices;
    const auto &ib_copy = mesh.indices;

    auto aabb_data_result = create_mesh_aabb_data(ctx.allocator, mesh, obj_path.filename().string());

    if (!aabb_data_result) {
        return tl::make_unexpected(aabb_data_result.error());
    }
    auto aabb_data = std::move(aabb_data_result.value());

    auto position_vb = mesh.vertices | std::views::transform([](const auto &v) {
                           return PositionOnlyVertex{
                                   .pos = v.position,
                                   .uvs = v.uvs,
                           };
                       }) |
                       to<std::vector<PositionOnlyVertex>>();

    auto vertex_buffer =
            Buffer::from_slice<Vertex>(ctx.allocator, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, std::span(vb_copy),
                                       std::format("vertex_buffer_{}", obj_path.filename().string()))
                    .value();

    auto pos_uv_buffer = Buffer::from_slice<PositionOnlyVertex>(
                                 ctx.allocator, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, std::span(position_vb),
                                 std::format("position_buffer_{}", obj_path.filename().string()))
                                 .value();

    auto index_buffer = Buffer::from_slice<u32>(ctx.allocator, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, std::span(ib_copy),
                                                std::format("index_buffer_{}", obj_path.filename().string()))
                                .value();

    auto indirect_cmds = mesh.submeshes | std::views::transform([](const auto &s) {
                             VkDrawIndexedIndirectCommand cmd{};
                             cmd.indexCount = s.index_count;
                             cmd.instanceCount = 1;
                             cmd.firstIndex = s.index_offset;
                             cmd.vertexOffset = 0;
                             cmd.firstInstance = 0;
                             return cmd;
                         }) |
                         to<std::vector<VkDrawIndexedIndirectCommand>>();

    const u32 pool_base = ctx.materials.register_batch(gpu_materials);

    return StaticMesh{
            .mesh = std::move(mesh),
            .indirect_template = {indirect_cmds.begin(), indirect_cmds.end()},
            .material_pool_base = pool_base,
            .material_count = static_cast<u32>(gpu_materials.size()),
            .vertex_buffer = ctx.create_buffer(std::move(vertex_buffer)),
            .pos_uv_buffer = ctx.create_buffer(std::move(pos_uv_buffer)),
            .index_buffer = ctx.create_buffer(std::move(index_buffer)),
            .draw_count = static_cast<u32>(indirect_cmds.size()),
            .mesh_aabb = aabb_data.mesh_aabb,
            .submesh_aabbs = std::move(aabb_data.submesh_aabbs),
            .aabb_buffer = ctx.create_buffer(std::move(aabb_data.device_buffer)),
    };
}


namespace {

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    auto read_raw(const std::filesystem::path &path) -> tl::expected<std::vector<std::byte>, Error> {
        std::ifstream f(path, std::ios::binary);
        if (!f)
            return tl::unexpected(
                    Error::make_error(Error::Type::MeshLoadError, "Cannot open scene file: " + path.string()));
        f.seekg(0, std::ios::end);
        const auto sz = static_cast<size_t>(f.tellg());
        f.seekg(0, std::ios::beg);

        constexpr size_t k_prefix = 16; // [magic u64][src_hash u64]
        if (sz <= k_prefix)
            return tl::unexpected(
                    Error::make_error(Error::Type::MeshLoadError, "Scene file too small: " + path.string()));

        f.seekg(static_cast<std::streamoff>(k_prefix), std::ios::beg);
        const size_t payload_sz = sz - k_prefix;
        std::vector<std::byte> buf(payload_sz);
        f.read(std::bit_cast<char *>(buf.data()), static_cast<std::streamsize>(payload_sz));
        if (!f)
            return tl::unexpected(Error::make_error(Error::Type::MeshLoadError, "Read error: " + path.string()));
        return buf;
    }

    auto bzip2_decompress(std::span<const std::byte> src) -> tl::expected<std::vector<std::byte>, Error> {
        // We don't know the original size, so grow until it fits.
        size_t dst_cap = src.size() * 8;
        for (int attempt = 0; attempt < 6; ++attempt) {
            std::vector<std::byte> dst(dst_cap);
            unsigned int dst_len = static_cast<unsigned int>(dst_cap);

            const int rc = BZ2_bzBuffToBuffDecompress(std::bit_cast<char *>(dst.data()), &dst_len,
                                                      const_cast<char *>(std::bit_cast<const char *>(src.data())),
                                                      static_cast<unsigned int>(src.size()),
                                                      /*small*/ 0, /*verbosity*/ 0);

            if (rc == BZ_OK) {
                dst.resize(dst_len);
                return dst;
            }
            if (rc == BZ_OUTBUFF_FULL) {
                dst_cap *= 4;
                continue;
            }
            return tl::unexpected(
                    Error::make_error(Error::Type::MeshLoadError, "bzip2 decompress failed: " + std::to_string(rc)));
        }
        return tl::unexpected(Error::make_error(Error::Type::MeshLoadError, "bzip2 output buffer never large enough"));
    }

    template<class T>
    auto blob_span(std::span<const std::byte> file, const Tooling::BlobRange &r) -> std::span<const T> {
        const size_t byte_off = static_cast<size_t>(r.offset);
        const size_t byte_sz = static_cast<size_t>(r.size);
        if (byte_off + byte_sz > file.size())
            return {};
        return {reinterpret_cast<const T *>(file.data() + byte_off), byte_sz / sizeof(T)};
    }

    auto string_at(std::span<const std::byte> string_blob, u32 offset) -> std::string_view {
        if (offset >= string_blob.size())
            return {};
        const char *p = reinterpret_cast<const char *>(string_blob.data() + offset);
        return std::string_view{p}; // null-terminated
    }

    // -------------------------------------------------------------------------
    // KTX2 -> LoadedTextureCpu  (mirrors load_ktx2_cpu_bc7 in Mesh.cxx)
    // -------------------------------------------------------------------------

    auto decode_ktx2_bytes(std::span<const std::byte> ktx_bytes, TextureLoadPacket::Type type,
                           TextureLoadPacket::Class tex_class, std::string_view debug_name) -> LoadedTextureCpu {

        auto make_error_tex = [&]() -> LoadedTextureCpu {
            LoadedTextureCpu out{};
            out.name = std::string(debug_name);
            out.type = type;
            out.texture_class = tex_class;
            out.width = out.height = 1;
            out.levels = 1;
            out.vk_format =
                    (type == TextureLoadPacket::Type::SRGB) ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
            out.data = {255, 0, 255, 255};
            out.level_offset = {0};
            out.level_size = {4};
            return out;
        };

        ktxTexture2 *ktx2 = nullptr;
        KTX_error_code rc = ktxTexture_CreateFromMemory(
                reinterpret_cast<const ktx_uint8_t *>(ktx_bytes.data()), static_cast<ktx_size_t>(ktx_bytes.size()),
                KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, reinterpret_cast<ktxTexture **>(&ktx2));

        if (rc != KTX_SUCCESS || !ktx2)
            return make_error_tex();

        // Check for normal map metadata (same lambda pattern as Mesh.cxx)
        bool is_normal_map = false;
        {
            char *value = nullptr;
            u32 length = 0;
            if (KTX_SUCCESS ==
                ktxHashList_FindValue(&ktx2->kvDataHead, "KTXwriterScParams", &length, (void **) &value)) {
                std::string_view params(value, length);
                is_normal_map = (params.find("--normal-mode") != std::string_view::npos);
            }
        }

        if (ktxTexture2_NeedsTranscoding(ktx2)) {
            rc = ktxTexture2_TranscodeBasis(ktx2, KTX_TTF_BC7_RGBA, 0);
            if (rc != KTX_SUCCESS) {
                ktxTexture2_Destroy(ktx2);
                return make_error_tex();
            }
        }

        LoadedTextureCpu out{};
        out.name = std::string(debug_name);
        out.type = type;
        out.texture_class = tex_class;
        out.width = static_cast<u32>(ktx2->baseWidth);
        out.height = static_cast<u32>(ktx2->baseHeight);
        out.levels = static_cast<u32>(ktx2->numLevels);

        if (ktxTexture2_NeedsTranscoding(ktx2) == KTX_FALSE) {
            // Already transcoded above, set format
            if (is_normal_map) {
                out.vk_format = VK_FORMAT_BC7_UNORM_BLOCK;
            } else {
                out.vk_format =
                        (type == TextureLoadPacket::Type::SRGB) ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
            }
        } else {
            out.vk_format = static_cast<VkFormat>(ktx2->vkFormat);
        }

        out.level_offset.resize(out.levels);
        out.level_size.resize(out.levels);
        u32 total = 0;

        for (u32 level = 0; level < out.levels; ++level) {
            ktx_size_t off = 0;
            auto level_size = ktxTexture_GetImageSize(reinterpret_cast<ktxTexture *>(ktx2), level);
            rc = ktxTexture_GetImageOffset(reinterpret_cast<ktxTexture *>(ktx2), level, 0, 0, &off);

            if (rc != KTX_SUCCESS || level_size == 0) {
                ktxTexture_Destroy(reinterpret_cast<ktxTexture *>(ktx2));
                return make_error_tex();
            }

            out.level_offset[level] = total;
            out.level_size[level] = static_cast<u32>(level_size);
            total += static_cast<u32>(level_size);
        }

        out.data.resize(total);
        const u8 *base = reinterpret_cast<const u8 *>(ktxTexture_GetData(reinterpret_cast<ktxTexture *>(ktx2)));

        for (u32 level = 0; level < out.levels; ++level) {
            ktx_size_t off = 0;
            (void) ktxTexture_GetImageOffset(reinterpret_cast<ktxTexture *>(ktx2), level, 0, 0, &off);
            std::memcpy(out.data.data() + out.level_offset[level], base + off, out.level_size[level]);
        }

        ktxTexture_Destroy(reinterpret_cast<ktxTexture *>(ktx2));
        return out;
    }

    // -------------------------------------------------------------------------
    // Derive TextureLoadPacket::Type / Class from GPUMaterialData flag context.
    // We check which map slot this texture index occupies to pick type/class.
    // Simpler approach: pass them in from the material loop directly.
    // -------------------------------------------------------------------------

    struct TexSlot {
        TextureLoadPacket::Type type;
        TextureLoadPacket::Class tex_class;
    };

    constexpr TexSlot k_albedo_slot = {TextureLoadPacket::Type::SRGB, TextureLoadPacket::Class::Albedo};
    constexpr TexSlot k_normal_slot = {TextureLoadPacket::Type::Linear, TextureLoadPacket::Class::Normal};
    constexpr TexSlot k_rough_slot = {TextureLoadPacket::Type::Linear, TextureLoadPacket::Class::Roughness};
    constexpr TexSlot k_metal_slot = {TextureLoadPacket::Type::Linear, TextureLoadPacket::Class::Metallic};
    constexpr TexSlot k_occlusion_slot = {TextureLoadPacket::Type::Linear, TextureLoadPacket::Class::Occlusion};
    constexpr TexSlot k_emissive_slot = {TextureLoadPacket::Type::SRGB, TextureLoadPacket::Class::Emissive};

} // namespace

// -----------------------------------------------------------------------------
// Public entry point
// -----------------------------------------------------------------------------

auto load_scene(RenderContext &ctx, const std::filesystem::path &scene_path, float scale)
        -> tl::expected<StaticMesh, Error> {
    ZoneScopedNC("Load scene", 0xFFAA00);
    NanoProfiler profiler(std::format("Load scene for '{}'", scene_path.filename().string()).c_str());

    using namespace std::string_view_literals;
    const auto ext = scene_path.extension().string();
    if (!matches(ext, ".scene.bz2"sv, ".scene"sv, ".bz2"sv, ".bzip2"sv)) {
        return tl::unexpected(Error::make_error(Error::Type::MeshLoadError, "Invalid scene file extension"));
    }

    // 1. Read compressed file bytes
    auto compressed = read_raw(scene_path);
    if (!compressed)
        return tl::unexpected(compressed.error());

    // 2. Bzip2 decompress
    auto file_bytes_exp = bzip2_decompress(*compressed);
    if (!file_bytes_exp)
        return tl::unexpected(file_bytes_exp.error());

    const std::vector<std::byte> file_bytes_owned = std::move(*file_bytes_exp);
    const std::span<const std::byte> file(file_bytes_owned);

    if (file.size() < sizeof(Tooling::FileHeader))
        return tl::unexpected(Error::make_error(Error::Type::MeshLoadError, "Scene file too small"));

    // 3. Validate header
    Tooling::FileHeader header{};
    std::memcpy(&header, file.data(), sizeof(header));

    if (header.magic != Tooling::k_magic)
        return tl::unexpected(Error::make_error(Error::Type::MeshLoadError, "Bad scene magic"));
    if (header.version != Tooling::k_version)
        return tl::unexpected(Error::make_error(Error::Type::MeshLoadError,
                                                "Unsupported scene version: " + std::to_string(header.version)));

    // 4. Fetch typed spans into the decompressed blob
    const auto file_submeshes = blob_span<Tooling::Submesh>(file, header.submesh_table);
    const auto file_vertices = blob_span<Tooling::Vertex>(file, header.vertex_blob);
    const auto file_indices = blob_span<u32>(file, header.index_blob);
    const auto file_materials = blob_span<Tooling::GPUMaterial>(file, header.material_table);
    const auto file_textures = blob_span<Tooling::Texture>(file, header.texture_table);
    const auto string_blob =
            file.subspan(static_cast<size_t>(header.string_blob.offset), static_cast<size_t>(header.string_blob.size));

    // -------------------------------------------------------------------------
    // 5. Determine which (type, class) each texture needs.
    //    We iterate materials once to record the first slot each texture index
    //    appears in. Last writer wins for dedup'd textures, but in practice
    //    the same image is never used as both albedo and normal.
    // -------------------------------------------------------------------------
    std::vector<TexSlot> tex_slots(header.texture_count, k_albedo_slot);
    {
        constexpr u32 k_none = 0xFFFFFFFFu;
        for (const auto &m: file_materials) {
            auto tag = [&](u32 idx, TexSlot slot) {
                if (idx != k_none && idx < header.texture_count)
                    tex_slots[idx] = slot;
            };
            tag(m.albedo_map, k_albedo_slot);
            tag(m.normal_map, k_normal_slot);
            tag(m.roughness_map, k_rough_slot);
            tag(m.metallic_map, k_metal_slot);
            tag(m.occlusion_map, k_occlusion_slot);
            tag(m.emissive_map, k_emissive_slot);
        }
    }

    // -------------------------------------------------------------------------
    // 6. Decode all KTX2 textures in parallel
    // -------------------------------------------------------------------------
    std::vector<std::future<LoadedTextureCpu>> tex_futures;
    tex_futures.reserve(header.texture_count);

    for (u32 i = 0; i < header.texture_count; ++i) {
        const Tooling::Texture &t = file_textures[i];

        const size_t ktx_off = static_cast<size_t>(t.ktx2_offset);
        const size_t ktx_sz = static_cast<size_t>(t.ktx2_size);

        if (ktx_off + ktx_sz > file.size()) {
            // Degenerate entry — push a placeholder future
            tex_futures.emplace_back(std::async(std::launch::deferred, []() -> LoadedTextureCpu {
                LoadedTextureCpu bad{};
                bad.width = bad.height = 1;
                bad.levels = 1;
                bad.vk_format = VK_FORMAT_R8G8B8A8_UNORM;
                bad.data = {255, 0, 255, 255};
                bad.level_offset = {0};
                bad.level_size = {4};
                return bad;
            }));
            continue;
        }

        // Copy the KTX2 bytes so the async lambda owns them.
        // (file_bytes_owned is referenced by the span but we can't safely share
        //  ownership here without extra machinery; a small copy is simplest.)
        std::vector<std::byte> ktx_copy(file.begin() + static_cast<ptrdiff_t>(ktx_off),
                                        file.begin() + static_cast<ptrdiff_t>(ktx_off + ktx_sz));

        const std::string debug_name = std::string(string_at(string_blob, t.name_str));
        const TexSlot slot = tex_slots[i];

        tex_futures.emplace_back(std::async(
                std::launch::async,
                [bytes = std::move(ktx_copy), slot, name = std::move(debug_name)]() mutable -> LoadedTextureCpu {
                    return decode_ktx2_bytes(std::span<const std::byte>(bytes), slot.type, slot.tex_class, name);
                }));
    }

    std::vector<LoadedTextureCpu> cpu_textures;
    cpu_textures.reserve(header.texture_count);
    for (auto &fut: tex_futures)
        cpu_textures.emplace_back(fut.get());

    // -------------------------------------------------------------------------
    // 7. Upload textures to GPU, collect TextureHandles
    // -------------------------------------------------------------------------
    std::vector<TextureHandle> tex_handles;
    tex_handles.reserve(cpu_textures.size());

    for (const auto &cpu_tex: cpu_textures) {
        auto img = create_texture_image_v2(ctx.allocator, *ctx.command_ctx, cpu_tex.width, cpu_tex.height,
                                           cpu_tex.vk_format, std::span<const u8>{cpu_tex.data},
                                           std::span<const u32>{cpu_tex.level_offset},
                                           std::span<const u32>{cpu_tex.level_size}, cpu_tex.name);
        tex_handles.emplace_back(ctx.textures.create(std::move(img)));
    }

    // -------------------------------------------------------------------------
    // 8. Convert file GPUMaterial -> runtime GPUMaterialData
    //    The file stores texture indices; we remap to bindless descriptor indices.
    // -------------------------------------------------------------------------
    const DefaultTextureHandles defs = get_default_texture_handles(ctx);
    constexpr u32 k_none = 0xFFFFFFFFu;

    auto remap_tex = [&](u32 file_idx, TextureHandle fallback) -> u32 {
        if (file_idx == k_none || file_idx >= static_cast<u32>(tex_handles.size()))
            return fallback.index();
        return tex_handles[file_idx].index();
    };

    std::vector<GPUMaterialData> gpu_materials;
    gpu_materials.reserve(file_materials.size());

    for (const auto &fm: file_materials) {
        GPUMaterialData out{};

        out.albedo_map = remap_tex(fm.albedo_map, defs.white);
        out.albedo_factor = {fm.albedo_factor[0], fm.albedo_factor[1], fm.albedo_factor[2], fm.albedo_factor[3]};
        out.set_albedo_map(out.albedo_map != defs.white.index());

        out.normal_map = remap_tex(fm.normal_map, defs.flat_normal);
        out.set_normal_map(out.normal_map != defs.flat_normal.index());

        out.roughness_map = remap_tex(fm.roughness_map, defs.white);
        out.roughness_factor = fm.roughness_factor;
        out.set_roughness_map(out.roughness_map != defs.white.index());

        out.metallic_map = remap_tex(fm.metallic_map, defs.black);
        out.metallic_factor = fm.metallic_factor;
        out.set_metallic_map(out.metallic_map != defs.black.index());

        out.occlusion_map = remap_tex(fm.occlusion_map, defs.white);
        out.set_occlusion_map(out.occlusion_map != defs.white.index());

        out.emissive_map = remap_tex(fm.emissive_map, defs.black);
        out.emissive_factor = {fm.emissive_factor[0], fm.emissive_factor[1], fm.emissive_factor[2]};
        out.set_emissive_map(out.emissive_map != defs.black.index());

        out.set_is_alpha_tested((fm.flags & GPUMaterialData::FLAG_ALPHA_TESTED) != 0);

        gpu_materials.emplace_back(out);
    }

    // -------------------------------------------------------------------------
    // 9. Convert file Vertex -> runtime Vertex
    //    File: float[3] position, float[2] uv0, u32 normal, u32 tangent, u32 reserved
    //    Runtime: glm::vec3 position, u32 uvs, u32 normal, u32 tangent, u32 bitangent
    //    Bitangent is absent in the file — we'll regenerate it below via
    //    compute_tangent_basis (same path as OBJ loading).
    // -------------------------------------------------------------------------
    MeshData mesh{};
    mesh.vertices.resize(file_vertices.size());

    for (size_t i = 0; i < file_vertices.size(); ++i) {
        const Tooling::Vertex &src = file_vertices[i];
        Vertex &dst = mesh.vertices[i];

        dst.position = {src.position[0] * scale, src.position[1] * scale, src.position[2] * scale};

        dst.uvs = src.uvs;
        dst.normal = src.normal;
        dst.tangent = src.tangent;

        const glm::vec4 t4 = glm::unpackSnorm3x10_1x2(src.tangent);
        const glm::vec3 n = safe_normalize(glm::vec3(glm::unpackSnorm3x10_1x2(src.normal)));
        const glm::vec3 t = safe_normalize(glm::vec3(t4));
        const float handedness = (t4.w < 0.0f) ? -1.0f : 1.0f;
        dst.bitangent = pack_dir(safe_normalize(glm::cross(n, t) * handedness));
    }
    mesh.indices.assign(file_indices.begin(), file_indices.end());

    // -------------------------------------------------------------------------
    // 10. Convert file Submesh -> runtime Submesh
    // -------------------------------------------------------------------------
    mesh.submeshes.reserve(file_submeshes.size());
    std::vector<u32> submesh_material_ids;
    submesh_material_ids.reserve(file_submeshes.size());

    for (const auto &fs: file_submeshes) {
        const u32 mat_idx = (fs.material_index < static_cast<u32>(gpu_materials.size())) ? fs.material_index : 0u;

        const bool alpha = (mat_idx < static_cast<u32>(file_materials.size()))
                                   ? ((file_materials[mat_idx].flags & GPUMaterialData::FLAG_ALPHA_TESTED) != 0)
                                   : false;

        mesh.submeshes.push_back(Submesh{
                .index_offset = fs.index_offset,
                .index_count = fs.index_count,
                .material_id = mat_idx,
                .alpha_tested = alpha,
        });
        submesh_material_ids.emplace_back(mat_idx);
    }

    // -------------------------------------------------------------------------
    // 11. Regenerate bitangents
    // Not necessary anymore since GLTF export now includes them.
    // -------------------------------------------------------------------------
    // compute_tangent_basis(mesh);

    // -------------------------------------------------------------------------
    // 12. AABB
    // -------------------------------------------------------------------------
    auto aabb_result = create_mesh_aabb_data(ctx.allocator, mesh, scene_path.filename().string());
    if (!aabb_result)
        return tl::unexpected(aabb_result.error());
    auto aabb_data = std::move(aabb_result.value());

    // -------------------------------------------------------------------------
    // 13. GPU buffer uploads
    // -------------------------------------------------------------------------
    const std::string stem = scene_path.stem().string();

    auto vertex_buffer = Buffer::from_slice<Vertex>(ctx.allocator, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                                    std::span<const Vertex>{mesh.vertices.data(), mesh.vertices.size()},
                                                    std::format("vertex_buffer_{}", stem))
                                 .value();


    auto position_vb =
            mesh.vertices |
            std::views::transform([](const Vertex &v) { return PositionOnlyVertex{.pos = v.position, .uvs = v.uvs}; }) |
            to<std::vector<PositionOnlyVertex>>();

    auto pos_uv_buffer = Buffer::from_slice<PositionOnlyVertex>(
                                 ctx.allocator, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                 std::span<const PositionOnlyVertex>{position_vb.data(), position_vb.size()},
                                 std::format("position_buffer_{}", stem))
                                 .value();

    auto index_buffer = Buffer::from_slice<u32>(ctx.allocator, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                                std::span<const u32>{mesh.indices.data(), mesh.indices.size()},
                                                std::format("index_buffer_{}", stem))
                                .value();

    auto indirect_cmds = mesh.submeshes | std::views::transform([](const Submesh &s) {
                             VkDrawIndexedIndirectCommand cmd{};
                             cmd.indexCount = s.index_count;
                             cmd.instanceCount = 1;
                             cmd.firstIndex = s.index_offset;
                             cmd.vertexOffset = 0;
                             cmd.firstInstance = 0;
                             return cmd;
                         }) |
                         to<std::vector<VkDrawIndexedIndirectCommand>>();

    const u32 pool_base = ctx.materials.register_batch(gpu_materials);

    return StaticMesh{
            .mesh = std::move(mesh),
            .indirect_template = {indirect_cmds.begin(), indirect_cmds.end()},
            .material_pool_base = pool_base,
            .material_count = static_cast<u32>(gpu_materials.size()),
            .vertex_buffer = ctx.buffers.create(std::move(vertex_buffer)),
            .pos_uv_buffer = ctx.buffers.create(std::move(pos_uv_buffer)),
            .index_buffer = ctx.buffers.create(std::move(index_buffer)),
            .draw_count = static_cast<u32>(indirect_cmds.size()),
            .mesh_aabb = aabb_data.mesh_aabb,
            .submesh_aabbs = std::move(aabb_data.submesh_aabbs),
            .aabb_buffer = ctx.buffers.create(std::move(aabb_data.device_buffer)),
    };
}
