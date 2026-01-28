#include "Mesh.hxx"

#include <glm/gtc/packing.hpp>

namespace {
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

        out.normal_map = resolve_texture(loaded, m.normal_map, defs.flat_normal);

        out.roughness_map = resolve_texture(loaded, m.roughness_map, defs.white);
        out.roughness_factor = m.roughness_factor;

        out.metallic_map = resolve_texture(loaded, m.metallic_map, defs.black);
        out.metallic_factor = m.metallic_factor;

        out.occlusion_map = resolve_texture(loaded, m.occlusion_map, defs.white);

        out.emissive_map = resolve_texture(loaded, m.emissive_map, defs.black);
        out.emissive_factor = m.emissive_factor;

        return out;
    }
} // namespace

auto load_mtl(const std::filesystem::path &mtl_path) -> std::unordered_map<std::string, MaterialData> {
    std::ifstream file{mtl_path};
    if (!file)
        return {};

    std::unordered_map<std::string, MaterialData> materials;
    MaterialData *current_material = nullptr;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "newmtl") {
            std::string name;
            iss >> name;
            current_material = &materials[name];
            current_material->name = name;
        } else if (current_material) {
            if (token == "map_Kd") {
                iss >> current_material->albedo_map;
            } else if (token == "Kd") {
                iss >> current_material->albedo_factor.r >> current_material->albedo_factor.g >>
                        current_material->albedo_factor.b;
                current_material->albedo_factor.a = 1.0f;
            } else if (token == "map_Bump" || token == "bump" || token == "norm") {
                iss >> current_material->normal_map;
            } else if (token == "map_Pr" || token == "map_Roughness") {
                iss >> current_material->roughness_map;
            } else if (token == "Pr") {
                iss >> current_material->roughness_factor;
            } else if (token == "map_Pm" || token == "map_Metallic") {
                iss >> current_material->metallic_map;
            } else if (token == "Pm") {
                iss >> current_material->metallic_factor;
            } else if (token == "map_Ka") {
                iss >> current_material->occlusion_map;
            } else if (token == "map_Ke") {
                iss >> current_material->emissive_map;
            } else if (token == "Ke") {
                iss >> current_material->emissive_factor.r >> current_material->emissive_factor.g >>
                        current_material->emissive_factor.b;
            } else if (token == "Ns") {
                float ns;
                iss >> ns;
                current_material->roughness_factor = 1.0f - std::sqrt(ns / 1000.0f);
            }
        }
    }

    return materials;
}

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
    std::ifstream file{obj_path};
    if (!file)
        return std::nullopt;

    struct VertexHash {
        size_t operator()(const Vertex &v) const {
            // better hashes exist; fine for now
            size_t h = 0;
            h ^= std::hash<float>()(v.position.x) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<float>()(v.position.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<float>()(v.position.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<u32>()(v.normal) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<u32>()(v.uvs) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };

    // OBJ temp arrays
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;

    // Output mesh (single pool)
    MeshData mesh{};
    std::unordered_map<Vertex, u32, VertexHash> vertex_map;

    // Materials
    std::unordered_map<std::string, MaterialData> materials;
    MaterialIdTable material_ids{};

    // Current submesh state
    std::string current_material_name = "default";
    u32 current_material_id = get_or_create_material_id(material_ids, current_material_name);
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

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#')
            continue;

        if (line.starts_with("mtllib ")) {
            std::istringstream iss(line.substr(7));
            std::string mtl_filename;
            iss >> mtl_filename;

            std::filesystem::path mtl_path = obj_path.parent_path() / mtl_filename;
            materials = load_mtl(mtl_path);

        } else if (line.starts_with("usemtl ")) {
            // finish previous submesh range
            flush_submesh();

            std::istringstream iss(line.substr(7));
            iss >> current_material_name;

            current_material_id = get_or_create_material_id(material_ids, current_material_name);

        } else if (line.starts_with("v ")) {
            glm::vec3 pos{};
            std::istringstream iss(line.substr(2));
            iss >> pos.x >> pos.y >> pos.z;
            positions.push_back(pos);

        } else if (line.starts_with("vn ")) {
            glm::vec3 n{};
            std::istringstream iss(line.substr(3));
            iss >> n.x >> n.y >> n.z;
            normals.push_back(n);

        } else if (line.starts_with("vt ")) {
            glm::vec2 uv{};
            std::istringstream iss(line.substr(3));
            iss >> uv.x >> uv.y;
            texcoords.push_back(uv);

        } else if (line.starts_with("f ")) {
            std::vector<std::array<int, 3>> face_vertices;

            std::istringstream iss(line.substr(2));
            std::string vertex_str;
            while (iss >> vertex_str) {
                std::array<int, 3> idx = {0, 0, 0}; // pos, uv, normal

                std::size_t first_slash = vertex_str.find('/');
                std::size_t second_slash = vertex_str.find('/', first_slash + 1);

                idx[0] = std::stoi(vertex_str.substr(0, first_slash));

                if (first_slash != std::string::npos) {
                    if (second_slash != std::string::npos) {
                        if (second_slash != first_slash + 1) {
                            idx[1] = std::stoi(vertex_str.substr(first_slash + 1, second_slash - first_slash - 1));
                        }
                        idx[2] = std::stoi(vertex_str.substr(second_slash + 1));
                    } else {
                        idx[1] = std::stoi(vertex_str.substr(first_slash + 1));
                    }
                }

                face_vertices.push_back(idx);
            }

            for (std::size_t i = 1; i + 1 < face_vertices.size(); ++i) {
                std::array<std::size_t, 3> tri = {0, i, i + 1};

                for (std::size_t j = 0; j < 3; ++j) {
                    const auto &vert_idx = face_vertices[tri[j]];

                    Vertex v{};
                    v.position = positions[static_cast<size_t>(vert_idx[0] - 1)];

                    glm::vec3 n = (vert_idx[2] > 0) ? normals[static_cast<size_t>(vert_idx[2] - 1)]
                                                    : glm::vec3(0.0f, 1.0f, 0.0f);

                    glm::vec2 uv =
                            (vert_idx[1] > 0) ? texcoords[static_cast<size_t>(vert_idx[1] - 1)] : glm::vec2(0.0f);

                    v.normal = glm::packSnorm3x10_1x2(glm::vec4{n, 0.0f});
                    v.uvs = glm::packUnorm4x8(glm::vec4{uv, 0.0f, 0.0f});

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
        }
    }

    // Flush final submesh
    flush_submesh();

    // Ensure at least one material exists
    if (materials.empty()) {
        materials["default"] = MaterialData{
                .name = "default",
                .albedo_map = "",
                .albedo_factor = glm::vec4{1.0f},
                .normal_map = "",
                .roughness_map = "",
                .roughness_factor = 1.0f,
                .metallic_map = "",
                .metallic_factor = 0.0f,
                .occlusion_map = "",
                .emissive_map = "",
                .emissive_factor = glm::vec3{0.0f},
        };
    }
    if (!materials.contains("default")) {
        // If OBJ references "default" but MTL didn't define it, add a sane fallback.
        materials["default"] = MaterialData{
                .name = "default",
                .albedo_factor = glm::vec4{1.0f},
                .roughness_factor = 1.0f,
                .metallic_factor = 0.0f,
                .emissive_factor = glm::vec3{0.0f},
        };
    }

    // -------------------------------------------------------------------------
    // Texture loading (dedupe) - same structure as your original
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

    // Create images + handles
    std::vector<TextureHandle> handles;
    handles.reserve(textures.size());

    for (const auto &tex: textures) {
        auto img = create_image_from_span_v2(ctx.allocator, cmd_ctx, tex.width, tex.height, tex.to_format(),
                                             std::span<const uint8_t>{tex.rgba.data(), tex.rgba.size()}, tex.name);
        handles.emplace_back(ctx.textures.create(std::move(img)));
    }

    // -------------------------------------------------------------------------
    // Build GPU materials in material_id order + upload buffer
    // -------------------------------------------------------------------------

    DefaultTextureHandles defs = get_default_texture_handles(ctx);
    LoadedTextureTable loaded = build_loaded_texture_table(textures, handles);

    std::vector<GPUMaterialData> gpu_materials;
    gpu_materials.reserve(material_ids.id_to_name.size());

    for (const std::string &mat_name: material_ids.id_to_name) {
        auto it = materials.find(mat_name);
        if (it == materials.end()) {
            // Missing material entry: fallback
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
                       std::ranges::to<std::vector<glm::vec3>>();

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
                             cmd.instanceCount = 1; // Placeholder
                             cmd.firstIndex = s.index_offset; // index element offset
                             cmd.vertexOffset = 0; // indices are global into mesh.vertices
                             cmd.firstInstance = 0; // Placeholder
                             return cmd;
                         }) |
                         std::ranges::to<std::vector<VkDrawIndexedIndirectCommand>>();
    ;
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
