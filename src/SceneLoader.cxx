// SceneLoader.cxx
#include "SceneLoader.hxx"

#include <algorithm>
#include <bit>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <glm/gtc/packing.hpp>
#include <iterator>
#include <numeric>
#include <optional>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>

#include "Assert.hxx"
#include "Compression.hxx"
#include "Logger.hxx"
#include "Material.hxx"
#include "Numeric.hxx"
#include "Stream.hxx"
#include "Types.hxx"

#include <meshoptimizer.h>

namespace Tooling {

    namespace {
        auto xdg_cache_scene_path(const std::filesystem::path &relative_output)
                -> std::optional<std::filesystem::path> {
            const char *xdg = std::getenv("XDG_CACHE_HOME");
            std::filesystem::path cache_root;
            if (xdg && xdg[0] != '\0') {
                cache_root = std::filesystem::path(xdg) / "bindless-headless" / "meshes";
            } else {
                const char *home = std::getenv("HOME");
                if (!home || home[0] == '\0')
                    return std::nullopt;
                cache_root = std::filesystem::path(home) / ".cache" / "bindless-headless" / "meshes";
            }
            return normalize_scene_out_path(cache_root / relative_output);
        }


        auto ensure_directory(const std::filesystem::path &path) -> bool {
            std::error_code ec;
            const bool created = std::filesystem::create_directory(path, ec);
            ASSERT(!ec && ec.message().c_str(), "Could not create directory, and did not exist");
            return created;
        }

        auto fnv1a_64(std::span<const std::byte> data) -> u64 {
            u64 hash = 0xcbf29ce484222325ULL;
            for (const auto b: data) {
                hash ^= static_cast<u64>(b);
                hash *= 0x00000100000001B3ULL;
            }
            return hash;
        }

        auto read_file_bytes(const std::filesystem::path &path) -> tl::expected<std::vector<std::byte>, Error> {
            std::ifstream f(path, std::ios::binary);
            if (!f)
                return tl::unexpected(make_error("Failed to open file for reading: " + path.string()));
            f.seekg(0, std::ios::end);
            const auto size = static_cast<size_t>(f.tellg());
            f.seekg(0, std::ios::beg);
            std::vector<std::byte> bytes(size);
            if (size > 0)
                f.read(std::bit_cast<char *>(bytes.data()), static_cast<std::streamsize>(size));
            if (!f && size > 0)
                return tl::unexpected(make_error("Failed to read file: " + path.string()));
            return bytes;
        }

        auto read_existing_src_hash(const std::filesystem::path &path) -> u64 {
            std::ifstream f(path, std::ios::binary);
            if (!f)
                return 0;
            u64 magic = 0, hash = 0;
            f.read(std::bit_cast<char *>(&magic), 8);
            f.read(std::bit_cast<char *>(&hash), 8);
            if (!f || magic != k_prefix_magic)
                return 0;
            return hash;
        }

        auto canonical_key(const std::filesystem::path &base_dir, std::string_view uri) -> std::string {
            std::filesystem::path p = base_dir / std::filesystem::path(std::string(uri));
            std::error_code ec;
            auto canon = std::filesystem::weakly_canonical(p, ec);
            if (ec)
                canon = p.lexically_normal();
            return canon.string();
        }

        auto resolve_ktx2_path(const std::filesystem::path &gltf_dir, std::string_view uri)
                -> tl::expected<std::filesystem::path, Error> {
            const std::filesystem::path orig(uri);
            const std::filesystem::path ktx2_name = std::filesystem::path(orig.stem()).replace_extension(".ktx2");

            const std::array candidates = {
                    gltf_dir / ktx2_name,
                    gltf_dir / orig.parent_path() / ktx2_name,
                    gltf_dir / "ktx2" / ktx2_name,
                    gltf_dir / "ktx" / ktx2_name,
                    gltf_dir / "textures" / ktx2_name,
                    gltf_dir / "textures" / "ktx2" / ktx2_name,
                    gltf_dir / "textures" / "ktx" / ktx2_name,
            };

            for (const auto &c: candidates)
                if (std::filesystem::exists(c))
                    return c;

            return tl::unexpected(make_error(
                    std::format("No KTX2 found for '{}' (searched {} locations)", uri, std::size(candidates))));
        }

        // -- Texture cache ----------------------------------------------------

        enum class TextureUsage : u32 {
            Albedo = 1 << 0,
            Normal = 1 << 1,
            Roughness = 1 << 2,
            Metallic = 1 << 3,
            Occlusion = 1 << 4,
            Emissive = 1 << 5,
            MetallicRoughnessCombined = 1 << 6,
        };

        struct TextureBuild {
            std::string original_path;
            std::string name;
            std::vector<std::byte> ktx2_bytes;
        };

        struct TextureCache {
            StringMap<u32> key_to_index;
            std::vector<TextureBuild> textures;
        };

        auto ensure_texture(TextureCache &cache, const std::string &key, TextureBuild build) -> u32 {
            if (auto it = cache.key_to_index.find(key); it != cache.key_to_index.end())
                return it->second;
            const auto idx = static_cast<u32>(cache.textures.size());
            cache.key_to_index.try_emplace(key, idx);
            cache.textures.emplace_back(std::move(build));
            return idx;
        }

        auto lookup_texture(const TextureCache &cache, const std::string &key) -> std::optional<u32> {
            if (auto it = cache.key_to_index.find(key); it != cache.key_to_index.end())
                return it->second;
            return std::nullopt;
        }

    } // namespace

    auto SceneLoader::convert_gltf(const std::filesystem::path &scene_path, const std::filesystem::path &output_path)
            -> tl::expected<void, Error> {

        const std::filesystem::path scene_abs = resolve_under(m_meshes_root, scene_path);
        const std::filesystem::path gltf_dir_abs = scene_abs.parent_path();
        const std::filesystem::path out_abs_no_normalize = resolve_under(m_meshes_root, output_path);
        const std::filesystem::path out_abs = normalize_scene_out_path(out_abs_no_normalize);

        auto src_bytes = read_file_bytes(scene_abs);
        if (!src_bytes)
            return tl::unexpected(make_error(std::format("Failed to read source for hashing: {}", scene_abs.string())));
        const u64 src_hash = fnv1a_64(*src_bytes);

        // -- Up-to-date check -------------------------------------------------
        if (std::filesystem::exists(out_abs)) {
            if (const u64 existing = read_existing_src_hash(out_abs); existing == src_hash) {
                trace("Scene up to date, skipping: {}", out_abs.string());
                return {};
            }
            trace("Scene hash mismatch, reconverting: {}", scene_abs.string());
        } else if (const auto cp = xdg_cache_scene_path(output_path); cp.has_value()) {
            if (std::filesystem::exists(*cp)) {
                if (const u64 h = read_existing_src_hash(*cp); h == src_hash) {
                    trace("Restoring from cache: {}", cp->string());
                    std::error_code ec;
                    std::filesystem::create_directories(out_abs.parent_path(), ec);
                    std::filesystem::copy_file(*cp, out_abs, std::filesystem::copy_options::overwrite_existing, ec);
                    if (!ec)
                        return {};
                    warn("Cache restore failed ({}), reconverting", ec.message());
                }
            }
        }

        trace("Converting scene: {}", scene_abs.filename().string());
        const auto t_total = std::chrono::steady_clock::now();

        // -- Parse glTF -------------------------------------------------------
        trace("  [1/4] parsing glTF...");
        const auto t_parse = std::chrono::steady_clock::now();

        fastgltf::Parser parser(fastgltf::Extensions::KHR_texture_transform |
                                fastgltf::Extensions::KHR_materials_unlit |
                                fastgltf::Extensions::KHR_materials_emissive_strength);

        auto loaded = fastgltf::GltfDataBuffer::FromPath(scene_abs);
        if (!loaded)
            return tl::unexpected(make_error(std::format("Failed to read glTF file: {}", scene_abs.string())));
        auto data = std::move(loaded.get());

        // LoadExternalImages intentionally omitted: we never decode raw images.
        // Keeping the URI variant intact lets us resolve the .ktx2 by filename.
        constexpr auto opts = fastgltf::Options::LoadExternalBuffers | fastgltf::Options::DecomposeNodeMatrices |
                              fastgltf::Options::GenerateMeshIndices;

        auto asset_exp = parser.loadGltfBinary(data, gltf_dir_abs, opts, fastgltf::Category::All);
        if (!asset_exp)
            asset_exp = parser.loadGltf(data, gltf_dir_abs, opts);
        if (!asset_exp)
            return tl::unexpected(make_error("fastgltf failed to load glTF"));

        fastgltf::Asset asset = std::move(asset_exp.get());
        trace("  [1/4] parsed in {}ms  ({} meshes, {} materials, {} images)",
              std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_parse).count(),
              asset.meshes.size(), asset.materials.size(), asset.images.size());

        // -- Helpers ----------------------------------------------------------

        std::vector<Submesh> submeshes;
        std::vector<Vertex> vertices;
        std::vector<u32> indices;
        std::vector<GPUMaterial> gpu_materials;
        TextureCache tex_cache;
        StringTable strings;

        auto get_image_debug_name = [&](u32 idx) -> std::string {
            const auto &img = asset.images[idx];
            if (!img.name.empty())
                return std::string(img.name);
            if (auto *u = std::get_if<fastgltf::sources::URI>(&img.data))
                if (u->uri.valid() && !u->uri.isDataUri())
                    return std::filesystem::path(u->uri.fspath()).filename().string();
            return std::format("image_{}", idx);
        };

        // Resolve, read, and deduplicate a KTX2 file for a given image slot.
        auto load_ktx2 = [&](u32 image_index) -> tl::expected<u32, Error> {
            if (image_index >= static_cast<u32>(asset.images.size()))
                return tl::unexpected(make_error(std::format("Invalid image index {}", image_index)));

            const auto &img = asset.images[image_index];
            const auto *uri_src = std::get_if<fastgltf::sources::URI>(&img.data);
            if (!uri_src || !uri_src->uri.valid() || uri_src->uri.isDataUri())
                return tl::unexpected(make_error(std::format("'{}': no file URI - only external KTX2 is supported",
                                                             get_image_debug_name(image_index))));

            const std::string uri_str = uri_src->uri.fspath().generic_string();
            const std::string key = canonical_key(gltf_dir_abs, uri_str);

            if (auto hit = lookup_texture(tex_cache, key))
                return *hit;

            auto ktx2_path = resolve_ktx2_path(gltf_dir_abs, uri_str);
            if (!ktx2_path)
                return tl::unexpected(ktx2_path.error());

            auto bytes = read_file_bytes(*ktx2_path);
            if (!bytes)
                return tl::unexpected(bytes.error());

            trace("    loaded {} ({} KB)", ktx2_path->filename().string(), bytes->size() / 1024);

            TextureBuild tb;
            tb.original_path = key;
            tb.name = get_image_debug_name(image_index);
            tb.ktx2_bytes = std::move(*bytes);
            return ensure_texture(tex_cache, key, std::move(tb));
        };

        auto resolve_tex_to_image = [&](const std::optional<fastgltf::TextureInfo> &ti) -> std::optional<u32> {
            if (!ti)
                return std::nullopt;
            const u32 tex_idx = static_cast<u32>(ti->textureIndex);
            if (tex_idx >= asset.textures.size())
                return std::nullopt;
            const auto &tex = asset.textures[tex_idx];
            if (!tex.imageIndex.has_value())
                return std::nullopt;
            return static_cast<u32>(*tex.imageIndex);
        };

        trace("  [2/4] loading KTX2 textures and building materials...");
        const auto t_mat = std::chrono::steady_clock::now();

        gpu_materials.reserve(asset.materials.size());
        for (u32 mi = 0; mi < static_cast<u32>(asset.materials.size()); ++mi) {
            const auto &m = asset.materials[mi];

            GPUMaterial out{};
            {
                auto &c = m.pbrData.baseColorFactor;
                out.albedo_factor = {c[0], c[1], c[2], c[3]};
            }
            out.roughness_factor = m.pbrData.roughnessFactor;
            out.metallic_factor = m.pbrData.metallicFactor;
            {
                auto &e = m.emissiveFactor;
                out.emissive_factor = {e[0], e[1], e[2]};
            }

            if (m.alphaMode == fastgltf::AlphaMode::Mask)
                out.flags |= MaterialFlags::AlphaTested;
            else if (m.alphaMode == fastgltf::AlphaMode::Blend)
                out.flags |= MaterialFlags::Transparent;

            if (auto &tr = m.transmission; tr != nullptr) {
                out.transmission_factor = tr->transmissionFactor;
                out.flags |= MaterialFlags::HasTransmission;
            }

            // Helper: load texture, warn on failure, return index or nullopt.
            auto try_load = [&](std::optional<u32> img_idx) -> std::optional<u32> {
                if (!img_idx)
                    return std::nullopt;
                auto result = load_ktx2(*img_idx);
                if (!result) {
                    warn("    skipping texture for '{}': {}", m.name.empty() ? "unnamed" : m.name,
                         result.error().message);
                    return std::nullopt;
                }
                return *result;
            };

            if (auto t = try_load(resolve_tex_to_image(m.pbrData.baseColorTexture))) {
                out.albedo_map = *t;
                out.flags |= MaterialFlags::Albedo;
            }

            if (m.normalTexture.has_value()) {
                const u32 tex_idx = static_cast<u32>(m.normalTexture->textureIndex);
                if (tex_idx < asset.textures.size() && asset.textures[tex_idx].imageIndex.has_value()) {
                    if (auto t = try_load(static_cast<u32>(*asset.textures[tex_idx].imageIndex))) {
                        out.normal_map = *t;
                        out.flags |= MaterialFlags::Normal;
                    }
                }
            }

            if (auto t = try_load(resolve_tex_to_image(m.pbrData.metallicRoughnessTexture))) {
                out.roughness_map = *t;
                out.metallic_map = *t;
                out.flags |= MaterialFlags::Roughness;
                out.flags |= MaterialFlags::Metallic;
            }

            if (m.occlusionTexture.has_value()) {
                const u32 tex_idx = static_cast<u32>(m.occlusionTexture->textureIndex);
                if (tex_idx < asset.textures.size() && asset.textures[tex_idx].imageIndex.has_value()) {
                    if (auto t = try_load(static_cast<u32>(*asset.textures[tex_idx].imageIndex))) {
                        out.occlusion_map = *t;
                        out.flags |= MaterialFlags::Occlusion;
                    }
                }
            }

            if (auto t = try_load(resolve_tex_to_image(m.emissiveTexture))) {
                out.emissive_map = *t;
                out.flags |= MaterialFlags::Emissive;
            }

            // Double sided?
            if (m.doubleSided) {
                out.flags |= MaterialFlags::DoubleSided;
                error("Material '{}' has double-sided flag set, but no occlusion map",
                      m.name.empty() ? "unnamed" : m.name);
            }

            if (out.flags == MaterialFlags::None) {
                trace("    material '{}' [{}] factor-only: "
                      "albedo({:.2f},{:.2f},{:.2f},{:.2f}) rough:{:.2f} metal:{:.2f}",
                      m.name.empty() ? "unnamed" : m.name, mi, out.albedo_factor.at(0), out.albedo_factor.at(1),
                      out.albedo_factor.at(2), out.albedo_factor.at(3), out.roughness_factor, out.metallic_factor);
            } else {
                trace("    material '{}' [{}] flags: {}", m.name.empty() ? "unnamed" : m.name, mi,
                      to_string(out.flags));
            }

            gpu_materials.emplace_back(out);
        }

        trace("  [2/4] done in {}ms  ({} unique textures)",
              std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_mat).count(),
              tex_cache.textures.size());

        // -- Meshes -----------------------------------------------------------
        constexpr std::array<float, k_lod_count> k_lod_ratios = {1.0f, 0.5f, 0.25f, 0.125f};
        auto to_glm = [](const fastgltf::math::mat<float, 4, 4> &m) { return glm::make_mat4(m.data()); };

        struct NodeProcessor {
            const fastgltf::Asset &asset;
            std::vector<Submesh> &submeshes;
            std::vector<Vertex> &vertices;
            std::vector<u32> &indices;
            const std::vector<GPUMaterial> &gpu_materials;
            const std::array<float, k_lod_count> &lod_ratios;
            decltype(to_glm) &to_glm_fn;

            void operator()(u32 node_idx, fastgltf::math::mat<float, 4, 4> parent_matrix) {
                const auto &node = asset.nodes[node_idx];
                const auto &global_mat = fastgltf::getTransformMatrix(node, parent_matrix);
                const glm::mat4 model_mat = to_glm_fn(global_mat);
                const glm::mat3 normal_mat = glm::transpose(glm::inverse(glm::mat3(model_mat)));

                if (node.meshIndex.has_value()) {
                    const auto &mesh = asset.meshes[*node.meshIndex];

                    for (const auto &prim: mesh.primitives) {
                        if (!prim.findAttribute("POSITION"))
                            continue;

                        auto get_attr = [&](std::string_view name) -> const fastgltf::Accessor * {
                            auto it = prim.findAttribute(name);
                            return (it == prim.attributes.cend()) ? nullptr : &asset.accessors[it->accessorIndex];
                        };

                        const auto *pos_acc = get_attr("POSITION");
                        const auto *uv_0_acc = get_attr("TEXCOORD_0");
                        const auto *uv_1_acc = get_attr("TEXCOORD_1");
                        const auto *nrm_acc = get_attr("NORMAL");
                        const auto *tan_acc = get_attr("TANGENT");

                        std::vector<glm::vec3> positions(pos_acc->count);
                        fastgltf::copyFromAccessor<glm::vec3>(asset, *pos_acc, positions.data());

                        std::vector<glm::vec2> uvs_0;
                        if (uv_0_acc) {
                            uvs_0.resize(uv_0_acc->count);
                            fastgltf::copyFromAccessor<glm::vec2>(asset, *uv_0_acc, uvs_0.data());
                        }

                        std::vector<glm::vec2> uvs_1;
                        if (uv_1_acc) {
                            uvs_1.resize(uv_1_acc->count);
                            fastgltf::copyFromAccessor<glm::vec2>(asset, *uv_1_acc, uvs_1.data());
                        }

                        std::vector<glm::vec3> normals;
                        if (nrm_acc) {
                            normals.resize(nrm_acc->count);
                            fastgltf::copyFromAccessor<glm::vec3>(asset, *nrm_acc, normals.data());
                        }

                        std::vector<glm::vec4> tangents;
                        if (tan_acc) {
                            tangents.resize(tan_acc->count);
                            fastgltf::copyFromAccessor<glm::vec4>(asset, *tan_acc, tangents.data());
                        }

                        std::vector<u32> local_indices;
                        if (prim.indicesAccessor.has_value()) {
                            const auto &idx_acc = asset.accessors[*prim.indicesAccessor];
                            local_indices.resize(idx_acc.count);
                            fastgltf::copyFromAccessor<u32>(asset, idx_acc, local_indices.data());
                        } else {
                            local_indices.resize(positions.size());
                            std::iota(local_indices.begin(), local_indices.end(), 0u);
                        }

                        u32 material_index =
                                prim.materialIndex.has_value() ? static_cast<u32>(*prim.materialIndex) : 0u;
                        if (material_index >= gpu_materials.size())
                            material_index = 0;

                        const auto vertex_offset = static_cast<u32>(vertices.size());
                        const size_t prim_vertex_count = positions.size();

                        std::vector<Vertex> local_verts(prim_vertex_count);
                        for (usize i = 0; i < prim_vertex_count; ++i) {
                            Vertex &v = local_verts[i];

                            const glm::vec4 wp = model_mat * glm::vec4(positions[i], 1.0f);
                            v.position = {wp.x, wp.y, wp.z};

                            v.uv0 = glm::packHalf2x16((i < uvs_0.size()) ? uvs_0[i] : glm::vec2{0.0f});
                            v.uv1 = glm::packHalf2x16((i < uvs_1.size()) ? uvs_1[i] : glm::vec2{0.0f});

                            glm::vec3 n = glm::normalize(normal_mat *
                                                         ((i < normals.size()) ? normals[i] : glm::vec3(0, 0, 1)));
                            v.normal = glm::packSnorm3x10_1x2(glm::vec4(n, 0.0f));

                            const glm::vec4 t4 = (i < tangents.size()) ? tangents[i] : glm::vec4(1, 0, 0, 1);
                            const glm::vec3 t3 = glm::normalize(glm::mat3(model_mat) * glm::vec3(t4));
                            v.tangent = glm::packSnorm3x10_1x2(glm::vec4(t3, t4.w));
                        }

                        std::vector<u32> opt_indices(local_indices.size());
                        std::vector<u32> remap(prim_vertex_count);
                        const usize unique_vertex_count =
                                meshopt_generateVertexRemap(remap.data(), local_indices.data(), local_indices.size(),
                                                            local_verts.data(), prim_vertex_count, sizeof(Vertex));

                        meshopt_remapIndexBuffer(opt_indices.data(), local_indices.data(), local_indices.size(),
                                                 remap.data());

                        std::vector<Vertex> remapped(unique_vertex_count);
                        meshopt_remapVertexBuffer(remapped.data(), local_verts.data(), prim_vertex_count,
                                                  sizeof(Vertex), remap.data());

                        meshopt_optimizeVertexCache(opt_indices.data(), opt_indices.data(), opt_indices.size(),
                                                    unique_vertex_count);
                        meshopt_optimizeOverdraw(opt_indices.data(), opt_indices.data(), opt_indices.size(),
                                                 &remapped[0].position[0], unique_vertex_count, sizeof(Vertex), 1.05f);

                        vertices.insert(vertices.end(), remapped.begin(), remapped.end());

                        Submesh sm{};
                        sm.vertex_offset = vertex_offset;
                        sm.vertex_count = static_cast<u32>(unique_vertex_count);
                        sm.material_index = material_index;

                        for (u32 lod = 0; lod < k_lod_count; ++lod) {
                            const u32 global_index_offset = static_cast<u32>(indices.size());
                            if (lod == 0) {
                                for (u32 idx: opt_indices)
                                    indices.push_back(vertex_offset + idx);
                                sm.lods[0] = {global_index_offset, static_cast<u32>(opt_indices.size())};
                            } else {
                                const size_t target =
                                        std::max(3u, static_cast<u32>(opt_indices.size() * lod_ratios[lod]));
                                std::vector<u32> simplified(opt_indices.size());
                                float err = 0.0f;
                                const size_t simplified_count =
                                        meshopt_simplify(simplified.data(), opt_indices.data(), opt_indices.size(),
                                                         &vertices[vertex_offset].position[0], unique_vertex_count,
                                                         sizeof(Vertex), target, 1e-2f, 0, &err);

                                simplified.resize(simplified_count);
                                meshopt_optimizeVertexCache(simplified.data(), simplified.data(), simplified_count,
                                                            unique_vertex_count);
                                for (u32 idx: simplified)
                                    indices.push_back(vertex_offset + idx);
                                sm.lods[lod] = {global_index_offset, static_cast<u32>(simplified_count)};
                            }
                        }
                        submeshes.emplace_back(sm);
                    }
                }

                for (u32 child_idx: node.children)
                    (*this)(child_idx, global_mat);
            }
        };

        trace("  [3/4] processing meshes...");
        const auto t_mesh = std::chrono::steady_clock::now();

        NodeProcessor proc{asset, submeshes, vertices, indices, gpu_materials, k_lod_ratios, to_glm};
        const auto &scene = asset.scenes[asset.defaultScene.value_or(0)];
        for (u32 root_node: scene.nodeIndices) {
            fastgltf::math::mat<float, 4, 4> identity{1.0F};
            proc(root_node, identity);
        }

        trace("  [3/4] done in {}ms  ({} submeshes, {} verts, {} indices)",
              std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_mesh).count(),
              submeshes.size(), vertices.size(), indices.size());

        // -- Serialise --------------------------------------------------------
        trace("  [4/4] serialising and compressing...");
        const auto t_serial = std::chrono::steady_clock::now();

        BinaryWriter w;
        FileHeader header{};
        header.submesh_count = static_cast<u32>(submeshes.size());
        header.vertex_count = static_cast<u32>(vertices.size());
        header.index_count = static_cast<u32>(indices.size());
        header.material_count = static_cast<u32>(gpu_materials.size());
        header.texture_count = static_cast<u32>(tex_cache.textures.size());
        header.content_hash = src_hash;

        const u64 header_offset = w.write_pod(header);

        w.align(k_align);
        header.submesh_table = w.write_pod_array<Submesh>(submeshes);
        w.align(k_align);
        header.vertex_blob = w.write_pod_array<Vertex>(vertices);
        w.align(k_align);
        header.index_blob = w.write_pod_array<u32>(indices);
        w.align(k_align);
        header.material_table = w.write_pod_array<GPUMaterial>(gpu_materials);

        w.align(k_align);
        const u64 texture_table_offset = w.size();
        {
            std::vector<Texture> placeholders(header.texture_count);
            header.texture_table = w.write_pod_array<Texture>(placeholders);
        }

        for (const auto &t: tex_cache.textures) {
            (void) strings.add(t.original_path);
            (void) strings.add(t.name);
        }

        w.align(k_align);
        header.string_blob = w.write_bytes(strings.blob());

        w.align(k_align);
        const u64 texture_blob_begin = w.size();

        for (u32 i = 0; i < header.texture_count; ++i) {
            w.align(k_align);
            const u64 ktx_off = w.size();
            const auto &tb = tex_cache.textures[i];
            const auto tex_range =
                    w.write_bytes(std::span<const std::byte>(tb.ktx2_bytes.data(), tb.ktx2_bytes.size()));

            Texture out{};
            out.original_path_str = strings.add(tb.original_path);
            out.name_str = strings.add(tb.name);
            out.ktx2_offset = ktx_off;
            out.ktx2_size = tex_range.size;
            w.patch_pod<Texture>(texture_table_offset + static_cast<u64>(i) * sizeof(Texture), out);
        }

        header.texture_blob.offset = texture_blob_begin;
        header.texture_blob.size = w.size() - texture_blob_begin;
        w.patch_pod<FileHeader>(header_offset, header);

        ensure_directory(out_abs_no_normalize.parent_path());

        auto compressed = scene_compress_to_memory(w.data(), src_hash);
        if (!compressed)
            return tl::unexpected(compressed.error());

        std::vector<std::filesystem::path> destinations = {out_abs_no_normalize};
        if (const auto cp = xdg_cache_scene_path(output_path); cp.has_value()) {
            std::error_code ec;
            std::filesystem::create_directories(cp->parent_path(), ec);
            if (!ec)
                destinations.push_back(*cp);
            else
                warn("Could not create cache dir, skipping: {}", cp->parent_path().string());
        }

        if (!write_scene_multi(*compressed, destinations))
            return tl::unexpected(make_error("Failed to write scene to one or more destinations"));

        trace("  [4/4] done in {}ms",
              std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_serial)
                      .count());
        trace("done: {} -> {} in {}ms", scene_abs.filename().string(), out_abs.filename().string(),
              std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_total)
                      .count());

        return {};
    }

} // namespace Tooling
