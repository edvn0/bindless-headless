#include "Mesh.hxx"
#include "CompilerGlue.hxx"

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

    auto unpack_normal(uint32_t packed) -> glm::vec3 {
        const glm::vec4 n4 = glm::unpackSnorm3x10_1x2(packed);
        return glm::vec3{n4.x, n4.y, n4.z};
    }

    auto pack_dir(glm::vec3 v) -> uint32_t { return glm::packSnorm3x10_1x2(glm::vec4{v, 0.0f}); }

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

            const auto uv0 = v0.uvs;
            const auto uv1 = v1.uvs;
            const auto uv2 = v2.uvs;

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
            h ^= std::hash<float>()(v.uvs.x) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<float>()(v.uvs.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
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
        v.uvs = uv0;
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

auto load_obj(RenderContext &ctx, GlobalCommandContext &cmd_ctx, const std::filesystem::path &obj_path)
        -> tl::expected<LoadedObj, Error> {

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
                create_texture_image_v2(ctx.allocator, cmd_ctx, tex.width, tex.height, tex.vk_format,
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

    auto material_buffer = Buffer::from_slice<GPUMaterialData>(
                                   ctx.allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                   std::span<const GPUMaterialData>{gpu_materials.data(), gpu_materials.size()},
                                   std::format("gpu_materials_{}", obj_path.filename().string()))
                                   .value();

    std::vector<u32> submesh_to_material_id_mapping;
    submesh_to_material_id_mapping.reserve(mesh.submeshes.size());
    for (const auto &submesh: mesh.submeshes) {
        submesh_to_material_id_mapping.emplace_back(submesh.material_id);
    }

    auto material_ids_buffer =
            Buffer::from_slice<u32>(ctx.allocator, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                    std::span(submesh_to_material_id_mapping),
                                    std::format("material_ids_buffer_{}", obj_path.filename().string()))
                    .value();

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

    indirect_cmds.reserve(mesh.submeshes.size());

    return LoadedObj{
            .mesh = std::move(mesh),
            .materials = std::move(materials),
            .gpu_materials = std::move(gpu_materials),
            .indirect_template = std::move(indirect_cmds),
            .material_buffer = ctx.buffers.create(std::move(material_buffer)),
            .material_ids_buffer = ctx.buffers.create(std::move(material_ids_buffer)),
            .vertex_buffer = ctx.buffers.create(std::move(vertex_buffer)),
            .pos_uv_buffer = ctx.buffers.create(std::move(pos_uv_buffer)),
            .index_buffer = ctx.buffers.create(std::move(index_buffer)),
            .draw_count = static_cast<u32>(indirect_cmds.size()),
            .mesh_aabb = aabb_data.mesh_aabb,
            .submesh_aabbs = std::move(aabb_data.submesh_aabbs),
            .aabb_buffer = ctx.buffers.create(std::move(aabb_data.device_buffer)),
    };
}
