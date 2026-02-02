#include "Mesh.hxx"
#include "CompilerGlue.hxx"

#include <glm/gtc/packing.hpp>

#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include <tiny_obj_loader.h>

namespace {
    static inline auto unpack_uv(uint32_t packed) -> glm::vec2 {
        const glm::vec4 u = glm::unpackUnorm4x8(packed);
        return glm::vec2{u.x, u.y};
    }

    static inline auto unpack_normal(uint32_t packed) -> glm::vec3 {
        const glm::vec4 n4 = glm::unpackSnorm3x10_1x2(packed);
        return glm::vec3{n4.x, n4.y, n4.z};
    }

    static inline auto pack_dir(glm::vec3 v) -> uint32_t {
        // v should be normalized (or close).
        return glm::packSnorm3x10_1x2(glm::vec4{v, 0.0f});
    }

    static inline auto safe_normalize(glm::vec3 v, float eps = 1e-20f) -> glm::vec3 {
        const float len2 = glm::dot(v, v);
        if (len2 <= eps)
            return glm::vec3{0.0f};
        return v * glm::inversesqrt(len2);
    }

    static inline auto build_any_orthonormal_tangent(glm::vec3 n) -> glm::vec3 {
        // Pick a vector not parallel to n, cross to get tangent.
        const glm::vec3 a = (std::abs(n.z) < 0.999f) ? glm::vec3{0.0f, 0.0f, 1.0f} : glm::vec3{0.0f, 1.0f, 0.0f};
        return safe_normalize(glm::cross(a, n));
    }

    static inline auto compute_tangent_basis(MeshData &mesh) -> void {
        if (mesh.vertices.empty() || mesh.indices.size() < 3)
            return;

        // If tangents already present (non-zero), skip.
        // (OBJ path: they'll be default-initialized to 0, so we compute.)
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

            const glm::vec2 uv0 = unpack_uv(v0.uvs);
            const glm::vec2 uv1 = unpack_uv(v1.uvs);
            const glm::vec2 uv2 = unpack_uv(v2.uvs);

            const glm::vec3 e1 = p1 - p0;
            const glm::vec3 e2 = p2 - p0;

            const glm::vec2 duv1 = uv1 - uv0;
            const glm::vec2 duv2 = uv2 - uv0;

            const float denom = duv1.x * duv2.y - duv1.y * duv2.x;
            if (std::abs(denom) <= eps) {
                // Degenerate UV mapping; skip contribution.
                continue;
            }

            const float r = 1.0f / denom;
            glm::vec3 t = (e1 * duv2.y - e2 * duv1.y) * r;
            glm::vec3 b = (e2 * duv1.x - e1 * duv2.x) * r;

            // Weight by triangle area (optional but helps a bit).
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

        // Orthonormalize per-vertex tangent, derive bitangent with handedness.
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
                // Gram-Schmidt: make t orthogonal to n
                t = safe_normalize(t - n * glm::dot(n, t));
                if (glm::dot(t, t) <= 0.0f)
                    t = build_any_orthonormal_tangent(n);
            }

            // Handedness from accumulated bitangent.
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

    auto build_loaded_texture_table(std::span<const TextureLoadPacket> textures, std::span<const TextureHandle> handles)
            -> LoadedTextureTable {
        LoadedTextureTable out{};
        out.by_name.reserve(textures.size());
        for (size_t i = 0; i < textures.size(); ++i) {
            out.by_name.emplace(textures[i].name, handles[i]);
        }
        return out;
    }

    auto resolve_texture(const LoadedTextureTable &loaded, const std::string &name, TextureHandle fallback) -> u32 {
        if (name.empty())
            return fallback.index();
        auto it = loaded.by_name.find(name);
        if (it != loaded.by_name.end())
            return it->second.index();
        return fallback.index();
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

        return out;
    }

    // tinyobj -> your MaterialData mapping
    auto to_material_data(const tinyobj::material_t &m) -> MaterialData {
        MaterialData out{};
        out.name = m.name;

        // Base color (Kd) and map_Kd
        out.albedo_factor = glm::vec4{m.diffuse[0], m.diffuse[1], m.diffuse[2], 1.0f};
        out.albedo_map = m.diffuse_texname;

        // Normal mapping: tinyobj uses bump_texname (and/or normal_texname in newer forks; tinyobjloader has bump)
        out.normal_map = m.bump_texname;

        // Roughness/metallic are not core OBJ/MTL, but some exporters stuff them into
        // "roughness_texname"/"metallic_texname" tinyobj::material_t has PBR fields in recent versions:
        // - roughness, metallic
        // - roughness_texname, metallic_texname
        // If your tinyobjloader version lacks these, this still compiles if you remove the block below.
#ifdef TINYOBJLOADER_USE_PBR_MATERIAL
        out.roughness_factor = m.roughness;
        out.metallic_factor = m.metallic;
        out.roughness_map = m.roughness_texname;
        out.metallic_map = m.metallic_texname;
#else
        // fallback from classic MTL: Ns (specular exponent) -> roughness (approx, same spirit as your old conversion)
        // tinyobj exposes shininess as "shininess"
        if (m.shininess > 0.0f) {
            out.roughness_factor = 1.0f - std::sqrt(m.shininess / 1000.0f);
        } else {
            out.roughness_factor = 1.0f;
        }
        out.metallic_factor = 0.0f;
#endif

        // Ambient map often used as AO (map_Ka)
        out.occlusion_map = m.ambient_texname;

        // Emissive (Ke + map_Ke)
        out.emissive_factor = glm::vec3{m.emission[0], m.emission[1], m.emission[2]};
        out.emissive_map = m.emissive_texname;

        // Sensible defaults if absent
        if (out.albedo_factor == glm::vec4(0.0f))
            out.albedo_factor = glm::vec4{1.0f};
        return out;
    }

    struct VertexHash {
        auto operator()(const Vertex &v) const noexcept -> size_t {
            // keep your original behavior for now (you already asked about improving it; can swap later)
            size_t h = 0;
            h ^= std::hash<float>()(v.position.x) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<float>()(v.position.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<float>()(v.position.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<u32>()(v.normal) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<u32>()(v.uvs) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };

    static inline auto pack_vertex_from_indices(const tinyobj::attrib_t &attrib, const tinyobj::index_t &idx)
            -> Vertex {
        Vertex v{};

        // positions (required by OBJ)
        if (idx.vertex_index >= 0) {
            const int base = 3 * idx.vertex_index;
            v.position = glm::vec3{
                    attrib.vertices[base + 0],
                    attrib.vertices[base + 1],
                    attrib.vertices[base + 2],
            };
        } else {
            v.position = glm::vec3{0.0f};
        }

        // normals (optional)
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

        // texcoords (optional)
        glm::vec2 uv0{0.0f};
        if (idx.texcoord_index >= 0 && !attrib.texcoords.empty()) {
            const int base = 2 * idx.texcoord_index;
            uv0 = glm::vec2{
                    attrib.texcoords[base + 0],
                    attrib.texcoords[base + 1],
            };
        }
        uv0.y = 1.0f - uv0.y; // flip V to match your pipeline
        v.uvs = glm::packUnorm4x8(glm::vec4{uv0, 0.0f, 0.0f});

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
    stbi_set_flip_vertically_on_load(true);

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
        -> std::optional<LoadedObj> {

    // -------------------------------------------------------------------------
    // Load OBJ + MTL via tinyobjloader
    // -------------------------------------------------------------------------
    tinyobj::ObjReaderConfig cfg{};
    cfg.mtl_search_path = obj_path.parent_path().string(); // where to find .mtl + textures
    cfg.triangulate = true; // you already triangulated manually; let tinyobj do it

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(obj_path.string(), cfg)) {
        // reader.Error() contains details, but you currently return nullopt on failure
        return std::nullopt;
    }

    const auto &attrib = reader.GetAttrib();
    const auto &shapes = reader.GetShapes();
    const auto &tiny_mats = reader.GetMaterials();

    // -------------------------------------------------------------------------
    // Output mesh (single pool) + dedupe
    // -------------------------------------------------------------------------
    MeshData mesh{};
    std::unordered_map<Vertex, u32, VertexHash> vertex_map;

    // -------------------------------------------------------------------------
    // Materials: tinyobj -> your MaterialData + your MaterialIdTable
    // -------------------------------------------------------------------------
    std::unordered_map<std::string, MaterialData> materials;
    materials.reserve(tiny_mats.size() + 1);

    MaterialIdTable material_ids{};

    // Always ensure a "default" material id exists (covers -1 material ids)
    const std::string default_name = "default";
    (void) get_or_create_material_id(material_ids, default_name);

    // Convert tinyobj materials
    for (const auto &m: tiny_mats) {
        MaterialData md = to_material_data(m);
        if (md.name.empty())
            md.name = default_name;

        (void) get_or_create_material_id(material_ids, md.name);
        materials.emplace(md.name, std::move(md));
    }

    // Ensure default material exists in your table
    if (!materials.contains(default_name)) {
        materials[default_name] = MaterialData{
                .name = default_name,
                .albedo_factor = glm::vec4{1.0f},
                .roughness_factor = 1.0f,
                .metallic_factor = 0.0f,
                .emissive_factor = glm::vec3{0.0f},
        };
    }

    // -------------------------------------------------------------------------
    // Submesh building: we create submeshes whenever material changes while
    // streaming triangles (like your old "usemtl" flush).
    // tinyobj provides per-face material_ids.
    // -------------------------------------------------------------------------
    u32 current_material_id = get_or_create_material_id(material_ids, default_name);
    u32 current_submesh_index_offset = 0;

    auto flush_submesh = [&]() {
        u32 index_count = static_cast<u32>(mesh.indices.size()) - current_submesh_index_offset;
        if (index_count == 0)
            return;

        mesh.submeshes.push_back(Submesh{
                .index_offset = current_submesh_index_offset,
                .index_count = index_count,
                .material_id = current_material_id,
        });
        current_submesh_index_offset = static_cast<u32>(mesh.indices.size());
    };

    for (const auto &shape: shapes) {
        size_t index_offset = 0;

        const auto &num_face_vertices = shape.mesh.num_face_vertices;
        const auto &material_ids_per_face = shape.mesh.material_ids;
        const auto &indices = shape.mesh.indices;

        for (size_t f = 0; f < num_face_vertices.size(); ++f) {
            const int fv = num_face_vertices[f];
            // cfg.triangulate=true => fv should be 3, but be defensive
            if (fv < 3) {
                index_offset += static_cast<size_t>(fv);
                continue;
            }

            const int mat_id = (f < material_ids_per_face.size()) ? material_ids_per_face[f] : -1;
            const std::string &mat_name =
                    (mat_id >= 0 && static_cast<size_t>(mat_id) < tiny_mats.size() && !tiny_mats[mat_id].name.empty())
                            ? tiny_mats[mat_id].name
                            : default_name;

            const u32 mat_u32 = get_or_create_material_id(material_ids, mat_name);

            if (mat_u32 != current_material_id) {
                flush_submesh();
                current_material_id = mat_u32;
            }

            // Triangulated faces are triangles, but if fv > 3, fan triangulate
            const tinyobj::index_t i0 = indices[index_offset + 0];

            for (int k = 1; k + 1 < fv; ++k) {
                const tinyobj::index_t i1 = indices[index_offset + static_cast<size_t>(k)];
                const tinyobj::index_t i2 = indices[index_offset + static_cast<size_t>(k + 1)];

                const tinyobj::index_t tri[3] = {i0, i1, i2};

                for (int j = 0; j < 3; ++j) {
                    Vertex v = pack_vertex_from_indices(attrib, tri[j]);

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

            index_offset += static_cast<size_t>(fv);
        }
    }

    flush_submesh();

    compute_tangent_basis(mesh);

    // -------------------------------------------------------------------------
    // Texture loading (dedupe) - unchanged, driven by your MaterialData table
    // -------------------------------------------------------------------------
    std::vector<std::future<TextureLoadPacket>> load_futures;
    std::unordered_set<std::string> unique_texture_names;
    const std::filesystem::path base_path = obj_path.parent_path();

#define LOAD_MAP(mat, field_name, t, clazz)                                                                            \
    do {                                                                                                               \
        if (!(mat).field_name.empty() && !unique_texture_names.contains((mat).field_name)) {                           \
            load_futures.emplace_back(                                                                                 \
                    std::async(std::launch::async, [base_path, tex_name = (mat).field_name]() -> TextureLoadPacket {   \
                        return load_texture_from_file(base_path / tex_name, TextureLoadPacket::Type::t,                \
                                                      TextureLoadPacket::Class::clazz);                                \
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

    std::vector<TextureLoadPacket> textures;
    textures.reserve(load_futures.size());
    for (auto &f: load_futures) {
        textures.emplace_back(f.get());
    }

    std::vector<TextureHandle> handles;
    handles.reserve(textures.size());

    for (const auto &tex: textures) {
        auto img = create_image_from_span_v2(ctx.allocator, cmd_ctx, tex.width, tex.height, tex.to_format(),
                                             std::span<const uint8_t>{tex.rgba.data(), tex.rgba.size()}, tex.name);
        handles.emplace_back(ctx.textures.create(std::move(img)));
    }

    // -------------------------------------------------------------------------
    // Build GPU materials in material_id order + upload buffer (unchanged)
    // -------------------------------------------------------------------------
    DefaultTextureHandles defs = get_default_texture_handles(ctx);
    LoadedTextureTable loaded = build_loaded_texture_table(textures, handles);

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

    auto position_vb = mesh.vertices | std::views::transform([](const auto &v) { return v.position; }) |
                       to<std::vector<glm::vec3>>();

    auto vertex_buffer =
            Buffer::from_slice<Vertex>(ctx.allocator, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, std::span(vb_copy),
                                       std::format("vertex_buffer_{}", obj_path.filename().string()))
                    .value();

    auto position_vertex_buffer =
            Buffer::from_slice<glm::vec3>(ctx.allocator, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, std::span(position_vb),
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
            .position_vertex_buffer = ctx.buffers.create(std::move(position_vertex_buffer)),
            .index_buffer = ctx.buffers.create(std::move(index_buffer)),
            .draw_count = static_cast<u32>(indirect_cmds.size()),
    };
}
