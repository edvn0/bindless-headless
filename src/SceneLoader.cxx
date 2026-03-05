// SceneLoader.cxx
#include "SceneLoader.hxx"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <glm/gtc/packing.hpp>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ktx.h>
#include <volk.h>

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>

#include "Logger.hxx"
#include "Material.hxx"
#include "Types.hxx"

#include <meshoptimizer.h>

#include <3PP/stb_image.h>
#include <3PP/stb_image_resize2.h>
#include <bzlib.h>

namespace Tooling {

    namespace {

        constexpr u64 k_align = 16;
        constexpr u64 k_prefix_magic = 0x454E4543534E5331ULL; // 'SNS1CENE'

        auto fnv1a_64(std::span<const std::byte> data) -> u64 {
            u64 hash = 0xcbf29ce484222325ULL;
            for (const auto b: data) {
                hash ^= static_cast<u64>(b);
                hash *= 0x00000100000001B3ULL;
            }
            return hash;
        }

        auto make_error(std::string msg) -> Error {
            return Error{.type = Error::Type::SceneLoaderError, .message = std::move(msg)};
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

            u64 magic = 0;
            u64 hash = 0;
            f.read(std::bit_cast<char *>(&magic), 8);
            f.read(std::bit_cast<char *>(&hash), 8);

            if (!f || magic != k_prefix_magic)
                return 0;

            return hash;
        }

        auto normalize_scene_out_path(std::filesystem::path out_path) -> std::filesystem::path {
            if (out_path.has_extension()) {
                out_path.replace_extension(".scene.bz2");
            } else {
                out_path += ".scene.bz2";
            }
            return out_path;
        }

        auto write_file_bytes_abs(const std::filesystem::path &out_path_no_normalize, std::span<std::byte> bytes,
                                  u64 src_hash) -> tl::expected<void, Error> {
            auto out_path = normalize_scene_out_path(out_path_no_normalize);
            info("Writing output scene file: {}", out_path.string());

            auto compressed_size = static_cast<unsigned int>(static_cast<float>(bytes.size()) * 1.01f + 600);
            std::vector<char> compressed(compressed_size);

            const int bz_rc =
                    BZ2_bzBuffToBuffCompress(compressed.data(), &compressed_size, std::bit_cast<char *>(bytes.data()),
                                             static_cast<unsigned int>(bytes.size()),
                                             /*blockSize100k*/ 9,
                                             /*verbosity*/ 0,
                                             /*workFactor*/ 0);

            if (bz_rc != BZ_OK)
                return tl::unexpected(make_error(std::format("BZ2_bzBuffToBuffCompress failed: {}", bz_rc)));

            std::ofstream f(out_path, std::ios::binary);
            if (!f)
                return tl::unexpected(make_error("Failed to open file for writing: " + out_path.string()));

            f.write(std::bit_cast<const char *>(&k_prefix_magic), 8);
            f.write(std::bit_cast<const char *>(&src_hash), 8);
            f.write(compressed.data(), static_cast<std::streamsize>(compressed_size));

            if (!f)
                return tl::unexpected(make_error("Failed to write file: " + out_path.string()));

            return {};
        }

        auto canonical_key(const std::filesystem::path &base_dir, std::string_view uri) -> std::string {
            std::filesystem::path p = base_dir / std::filesystem::path(std::string(uri));
            std::error_code ec;
            auto canon = std::filesystem::weakly_canonical(p, ec);
            if (ec)
                canon = p.lexically_normal();
            return canon.string();
        }

        enum class TextureUsage : u32 {
            Albedo = 1 << 0,
            Normal = 1 << 1,
            Roughness = 1 << 2,
            Metallic = 1 << 3,
            Occlusion = 1 << 4,
            Emissive = 1 << 5,
            MetallicRoughnessCombined = 1 << 6,
        };

        enum class OETF {
            Linear,
            sRGB,
        };

        auto texture_usage_to_oetf(TextureUsage usage) -> std::optional<OETF> {
            switch (usage) {
                using enum OETF;
                using enum TextureUsage;
                case Albedo:
                    return sRGB;
                case Normal:
                    return Linear;
                case Roughness:
                    return Linear;
                case Metallic:
                    return Linear;
                case Occlusion:
                    return Linear;
                case Emissive:
                    return sRGB;
                default:
                    return std::nullopt;
            }
        }

        struct LoadedImageRGBA {
            TextureUsage usage{};
            int width{};
            int height{};
            auto oetf() const { return texture_usage_to_oetf(usage); }
            std::vector<u8> rgba{};
        };

        auto decode_image_rgba8(std::span<const std::byte> bytes, TextureUsage usage)
                -> tl::expected<LoadedImageRGBA, Error> {
            int w = 0;
            int h = 0;
            int comp = 0;
            stbi_uc *data = stbi_load_from_memory(std::bit_cast<const stbi_uc *>(bytes.data()),
                                                  static_cast<int>(bytes.size()), &w, &h, &comp, 4);

            if (!data)
                return tl::unexpected(make_error(std::string("stb_image failed: ") + stbi_failure_reason()));

            LoadedImageRGBA out;
            out.width = w;
            out.height = h;
            out.usage = usage;
            out.rgba.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 4);
            std::memcpy(out.rgba.data(), data, out.rgba.size());
            stbi_image_free(data);
            return out;
        }

        auto make_ktx2_from_rgba8(const LoadedImageRGBA &img, bool generate_mips, bool basis_u_supercompression)
                -> tl::expected<std::vector<std::byte>, Error> {
            if (img.width <= 0 || img.height <= 0 || img.rgba.empty())
                return tl::unexpected(make_error("Invalid image data"));

            ktxTextureCreateInfo ci{};
            ci.vkFormat = VK_FORMAT_R8G8B8A8_UNORM;
            ci.baseWidth = static_cast<ktx_uint32_t>(img.width);
            ci.baseHeight = static_cast<ktx_uint32_t>(img.height);
            ci.baseDepth = 1;
            ci.numDimensions = 2;

            auto calculate_mips = [](int w, int h) {
                int levels = 1;
                while (w > 1 || h > 1) {
                    w = std::max(1, w / 2);
                    h = std::max(1, h / 2);
                    levels++;
                }
                return levels;
            };

            ci.numLevels = generate_mips ? static_cast<ktx_uint32_t>(calculate_mips(img.width, img.height)) : 1;
            ci.numLayers = 1;
            ci.numFaces = 1;
            ci.isArray = KTX_FALSE;

            ktxTexture2 *ktx2 = nullptr;
            KTX_error_code rc = ktxTexture2_Create(&ci, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &ktx2);
            if (rc != KTX_SUCCESS)
                return tl::unexpected(make_error("ktxTexture2_Create failed"));

            rc = ktxTexture_SetImageFromMemory(reinterpret_cast<ktxTexture *>(ktx2), 0, 0, 0, img.rgba.data(),
                                               static_cast<ktx_size_t>(img.rgba.size()));

            if (rc != KTX_SUCCESS) {
                ktxTexture_Destroy(reinterpret_cast<ktxTexture *>(ktx2));
                return tl::unexpected(make_error("ktxTexture_SetImageFromMemory failed"));
            }

            if (generate_mips) {
                std::vector<unsigned char> prev_level_data(img.rgba.begin(), img.rgba.end());
                int prev_w = img.width;
                int prev_h = img.height;

                const int num_levels = static_cast<int>(ci.numLevels);
                for (int level = 1; level < num_levels; ++level) {
                    const int mip_w = std::max(1, prev_w / 2);
                    const int mip_h = std::max(1, prev_h / 2);

                    std::vector<unsigned char> mip_data(mip_w * mip_h * 4);

                    const unsigned char *result = stbir_resize_uint8_linear(
                            prev_level_data.data(), prev_w, prev_h, 0, mip_data.data(), mip_w, mip_h, 0, STBIR_RGBA);

                    if (result == nullptr) {
                        ktxTexture2_Destroy(ktx2);
                        return tl::unexpected(make_error("stbir_resize_uint8_linear failed"));
                    }

                    rc = ktxTexture_SetImageFromMemory(reinterpret_cast<ktxTexture *>(ktx2),
                                                       static_cast<ktx_uint32_t>(level), 0, 0, mip_data.data(),
                                                       static_cast<ktx_size_t>(mip_data.size()));
                    if (rc != KTX_SUCCESS) {
                        ktxTexture2_Destroy(ktx2);
                        return tl::unexpected(make_error("ktxTexture_SetImageFromMemory failed for mip level"));
                    }

                    prev_level_data = std::move(mip_data);
                    prev_w = mip_w;
                    prev_h = mip_h;
                }
            }

            if (basis_u_supercompression) {
                ktxBasisParams params{};
                params.structSize = sizeof(params);
                params.uastc = KTX_TRUE;
                params.qualityLevel = 255;
                params.normalMap = (img.usage == TextureUsage::Normal) ? KTX_TRUE : KTX_FALSE;
                params.compressionLevel = 5;
                params.threadCount = std::max(1u, std::thread::hardware_concurrency() / 2);

                rc = ktxTexture2_CompressBasisEx(ktx2, &params);
                if (rc != KTX_SUCCESS) {
                    ktxTexture_Destroy(reinterpret_cast<ktxTexture *>(ktx2));
                    return tl::unexpected(make_error("ktxTexture2_CompressBasisEx failed (BasisU not enabled?)"));
                }
            }

            ktx_uint8_t *out_bytes = nullptr;
            ktx_size_t out_size = 0;
            rc = ktxTexture_WriteToMemory(reinterpret_cast<ktxTexture *>(ktx2), &out_bytes, &out_size);
            if (rc != KTX_SUCCESS) {
                ktxTexture_Destroy(reinterpret_cast<ktxTexture *>(ktx2));
                return tl::unexpected(make_error("ktxTexture_WriteToMemory failed"));
            }

            std::vector<std::byte> result(out_size);
            std::memcpy(result.data(), out_bytes, out_size);

            ktxTexture_Destroy(reinterpret_cast<ktxTexture *>(ktx2));
            return result;
        }

        auto get_gltf_image_bytes(const fastgltf::Asset &asset, const fastgltf::Image &image,
                                  const std::filesystem::path &gltf_dir_abs)
                -> tl::expected<std::vector<std::byte>, Error> {

            if (auto *uri = std::get_if<fastgltf::sources::URI>(&image.data)) {
                if (!uri->uri.valid())
                    return tl::unexpected(make_error("Image has empty URI"));

                const auto img_path = gltf_dir_abs / uri->uri.fspath();
                auto bytes = read_file_bytes(img_path);
                if (!bytes)
                    return tl::unexpected(bytes.error());
                return *bytes;
            }

            if (auto *bv = std::get_if<fastgltf::sources::BufferView>(&image.data)) {
                const auto &view = asset.bufferViews[bv->bufferViewIndex];
                const auto &buffer = asset.buffers[view.bufferIndex];

                auto get_buffer_ptr =
                        [&](const fastgltf::Buffer &buf) -> tl::expected<std::span<const std::byte>, Error> {
                    if (auto *vec = std::get_if<fastgltf::sources::Vector>(&buf.data)) {
                        return std::span<const std::byte>(reinterpret_cast<const std::byte *>(vec->bytes.data()),
                                                          vec->bytes.size());
                    }
                    if (auto *u = std::get_if<fastgltf::sources::URI>(&buf.data)) {
                        (void) u;
                        return tl::unexpected(
                                make_error("BufferView points to external buffer; handle by pre-loading buffers"));
                    }
                    return tl::unexpected(make_error("Unsupported buffer source"));
                };

                auto bufspan = get_buffer_ptr(buffer);
                if (!bufspan)
                    return tl::unexpected(bufspan.error());

                const size_t start = static_cast<size_t>(view.byteOffset);
                const size_t len = static_cast<size_t>(view.byteLength);
                if (start + len > bufspan->size())
                    return tl::unexpected(make_error("BufferView out of range"));

                std::vector<std::byte> bytes(len);
                std::memcpy(bytes.data(), bufspan->data() + start, len);
                return bytes;
            }

            if (auto *array = std::get_if<fastgltf::sources::Array>(&image.data)) {
                std::vector<std::byte> bytes(array->bytes.size());
                std::memcpy(bytes.data(), array->bytes.data(), array->bytes.size());
                return bytes;
            }

            if (auto *vector = std::get_if<fastgltf::sources::Vector>(&image.data)) {
                std::vector<std::byte> bytes(vector->bytes.size());
                std::memcpy(bytes.data(), vector->bytes.data(), vector->bytes.size());
                return bytes;
            }

            return tl::unexpected(make_error("Unsupported glTF image source"));
        }

        struct TextureBuild {
            std::string original_path;
            std::string name;
            std::vector<std::byte> ktx2_bytes;
        };

        struct TextureCache {
            std::unordered_map<std::string, u32, string_hash, string_eq> key_to_index;
            std::vector<TextureBuild> textures;
        };

        auto ensure_texture(TextureCache &cache, const std::string &key, TextureBuild build) -> u32 {
            if (auto it = cache.key_to_index.find(key); it != cache.key_to_index.end())
                return it->second;

            const auto idx = static_cast<u32>(cache.textures.size());
            std::ignore = cache.key_to_index.try_emplace(key, idx);
            std::ignore = cache.textures.emplace_back(std::move(build));
            return idx;
        }

    } // namespace

    auto SceneLoader::convert_gltf(const std::filesystem::path &scene_path, const std::filesystem::path &output_path)
            -> tl::expected<void, Error> {

        // --- Path model (single source of truth) ---
        // m_meshes_root is the root for mesh assets, e.g. "assets/meshes".
        // scene_path is relative to that root, e.g. "myMesh/myMesh.gltf".
        const std::filesystem::path scene_abs = resolve_under(m_meshes_root, scene_path);
        const std::filesystem::path gltf_dir_abs = scene_abs.parent_path();

        // Output path is relative to meshes_root by default (unless absolute).
        const std::filesystem::path out_abs_no_normalize = resolve_under(m_meshes_root, output_path);
        const std::filesystem::path out_abs = normalize_scene_out_path(out_abs_no_normalize);

        // Hash the source glTF file bytes for up-to-date skipping.
        auto src_bytes = read_file_bytes(scene_abs);
        if (!src_bytes)
            return tl::unexpected(make_error(std::format("Failed to read source for hashing: {}", scene_abs.string())));
        const u64 src_hash = fnv1a_64(*src_bytes);

        // Early-out if output exists and hash matches.
        if (std::filesystem::exists(out_abs)) {
            if (const u64 existing_hash = read_existing_src_hash(out_abs); existing_hash == src_hash) {
                info("Scene up to date, skipping conversion: {}", out_abs.string());
                return {};
            }
            info("Scene hash mismatch, reconverting: {}", scene_abs.string());
        }

        fastgltf::Parser parser(fastgltf::Extensions::KHR_texture_transform |
                                fastgltf::Extensions::KHR_materials_unlit |
                                fastgltf::Extensions::KHR_materials_emissive_strength);

        auto loaded = fastgltf::GltfDataBuffer::FromPath(scene_abs);
        if (!loaded)
            return tl::unexpected(make_error(std::format("Failed to read glTF file: {}", scene_abs.string())));
        auto data = std::move(loaded.get());

        constexpr auto opts = fastgltf::Options::LoadExternalBuffers | fastgltf::Options::LoadExternalImages |
                              fastgltf::Options::DecomposeNodeMatrices | fastgltf::Options::GenerateMeshIndices;

        // IMPORTANT: glTF base dir must be the directory containing the glTF file.
        auto asset_exp = parser.loadGltfBinary(data, gltf_dir_abs, opts, fastgltf::Category::All);
        if (!asset_exp) {
            asset_exp = parser.loadGltf(data, gltf_dir_abs, opts);
        }
        if (!asset_exp)
            return tl::unexpected(make_error("fastgltf failed to load glTF"));

        fastgltf::Asset asset = std::move(asset_exp.get());

        std::vector<Submesh> submeshes;
        std::vector<Vertex> vertices;
        std::vector<u32> indices;
        std::vector<GPUMaterial> gpu_materials;
        TextureCache tex_cache;
        StringTable strings;

        auto convert_image_to_ktx2 = [&](u32 image_index, std::string debug_uri_name,
                                         TextureUsage usage) -> tl::expected<u32, Error> {
            if (image_index >= asset.images.size())
                return tl::unexpected(make_error("Invalid image index"));

            const fastgltf::Image &img = asset.images[image_index];

            std::string key = std::format("image_index:{}", image_index);
            if (auto *uri = std::get_if<fastgltf::sources::URI>(&img.data)) {
                if (uri->uri.valid() && !uri->uri.isDataUri())
                    key = canonical_key(gltf_dir_abs, uri->uri.fspath().generic_string());
            }

            if (auto it = tex_cache.key_to_index.find(key); it != tex_cache.key_to_index.end())
                return it->second;

            auto bytes_exp = get_gltf_image_bytes(asset, img, gltf_dir_abs);
            if (!bytes_exp)
                return tl::unexpected(bytes_exp.error());

            auto decoded_exp = decode_image_rgba8(*bytes_exp, usage);
            if (!decoded_exp)
                return tl::unexpected(decoded_exp.error());

            const bool gen_mips = true;
            const bool use_basis = true;

            auto ktx2_exp = make_ktx2_from_rgba8(*decoded_exp, gen_mips, use_basis);
            if (!ktx2_exp)
                return tl::unexpected(ktx2_exp.error());

            TextureBuild tb{};
            tb.original_path = key;
            tb.name = std::move(debug_uri_name);
            tb.ktx2_bytes = std::move(*ktx2_exp);

            return ensure_texture(tex_cache, key, std::move(tb));
        };

        gpu_materials.reserve(asset.materials.size());
        for (u32 mi = 0; mi < static_cast<u32>(asset.materials.size()); ++mi) {
            const auto &m = asset.materials[mi];

            GPUMaterial out{};

            {
                const auto &c = m.pbrData.baseColorFactor;
                out.albedo_factor = {c[0], c[1], c[2], c[3]};
            }
            out.roughness_factor = m.pbrData.roughnessFactor;
            out.metallic_factor = m.pbrData.metallicFactor;

            {
                const auto &e = m.emissiveFactor;
                out.emissive_factor = {e[0], e[1], e[2]};
            }

            if (m.alphaMode == fastgltf::AlphaMode::Mask || m.alphaMode == fastgltf::AlphaMode::Blend) {
                out.flags |= GPUMaterialData::FLAG_ALPHA_TESTED;
            }

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

            if (auto img_idx = resolve_tex_to_image(m.pbrData.baseColorTexture); img_idx.has_value()) {
                auto texp = convert_image_to_ktx2(*img_idx, "baseColor", TextureUsage::Albedo);
                if (!texp)
                    return tl::unexpected(texp.error());
                out.albedo_map = *texp;
                out.flags |= GPUMaterialData::FLAG_ALBEDO_MAP;
            }

            if (m.normalTexture.has_value()) {
                const u32 tex_idx = static_cast<u32>(m.normalTexture->textureIndex);
                if (tex_idx < asset.textures.size() && asset.textures[tex_idx].imageIndex.has_value()) {
                    auto texp = convert_image_to_ktx2(static_cast<u32>(*asset.textures[tex_idx].imageIndex), "normal",
                                                      TextureUsage::Normal);
                    if (!texp)
                        return tl::unexpected(texp.error());
                    out.normal_map = *texp;
                    out.flags |= GPUMaterialData::FLAG_NORMAL_MAP;
                }
            }

            if (auto img_idx = resolve_tex_to_image(m.pbrData.metallicRoughnessTexture); img_idx.has_value()) {
                auto texp = convert_image_to_ktx2(*img_idx, "metalRough", TextureUsage::MetallicRoughnessCombined);
                if (!texp)
                    return tl::unexpected(texp.error());
                out.roughness_map = *texp;
                out.metallic_map = *texp;
                out.flags |= GPUMaterialData::FLAG_ROUGHNESS_MAP;
                out.flags |= GPUMaterialData::FLAG_METALLIC_MAP;
            }

            if (m.occlusionTexture.has_value()) {
                const auto tex_idx = static_cast<u32>(m.occlusionTexture->textureIndex);
                if (tex_idx < asset.textures.size() && asset.textures[tex_idx].imageIndex.has_value()) {
                    auto texp = convert_image_to_ktx2(static_cast<u32>(*asset.textures[tex_idx].imageIndex),
                                                      "occlusion", TextureUsage::Occlusion);
                    if (!texp)
                        return tl::unexpected(texp.error());
                    out.occlusion_map = *texp;
                    out.flags |= GPUMaterialData::FLAG_OCCLUSION_MAP;
                }
            }

            if (auto img_idx = resolve_tex_to_image(m.emissiveTexture); img_idx.has_value()) {
                auto texp = convert_image_to_ktx2(*img_idx, "emissive", TextureUsage::Emissive);
                if (!texp)
                    return tl::unexpected(texp.error());
                out.emissive_map = *texp;
                out.flags |= GPUMaterialData::FLAG_EMISSIVE_MAP;
            }

            gpu_materials.emplace_back(out);
        }

        for (const auto &mesh: asset.meshes) {
            for (const auto &prim: mesh.primitives) {
                if (!prim.findAttribute("POSITION"))
                    continue;

                auto get_attr = [&](std::string_view name) -> fastgltf::Accessor * {
                    auto it = prim.findAttribute(name);
                    if (it == prim.attributes.cend())
                        return nullptr;
                    return &asset.accessors[it->accessorIndex];
                };

                const auto *pos_acc = get_attr("POSITION");
                const auto *uv_acc = get_attr("TEXCOORD_0");
                const auto *nrm_acc = get_attr("NORMAL");
                const auto *tan_acc = get_attr("TANGENT");

                std::vector<glm::vec3> positions;
                std::vector<glm::vec2> uvs;
                std::vector<glm::vec3> normals;
                std::vector<glm::vec4> tangents;

                positions.resize(pos_acc->count);
                fastgltf::copyFromAccessor<glm::vec3>(asset, *pos_acc, positions.data());

                if (uv_acc) {
                    uvs.resize(uv_acc->count);
                    fastgltf::copyFromAccessor<glm::vec2>(asset, *uv_acc, uvs.data());
                }
                if (nrm_acc) {
                    normals.resize(nrm_acc->count);
                    fastgltf::copyFromAccessor<glm::vec3>(asset, *nrm_acc, normals.data());
                }
                if (tan_acc) {
                    tangents.resize(tan_acc->count);
                    fastgltf::copyFromAccessor<glm::vec4>(asset, *tan_acc, tangents.data());
                }

                std::vector<u32> local_indices;
                if (prim.indicesAccessor.has_value()) {
                    const auto &idx_acc = asset.accessors[*prim.indicesAccessor];
                    local_indices.resize(idx_acc.count);
                    fastgltf::copyFromAccessor<glm::u32>(asset, idx_acc, local_indices.data());
                } else {
                    local_indices.resize(positions.size());
                    std::iota(local_indices.begin(), local_indices.end(), 0);
                }

                u32 material_index = 0;
                if (prim.materialIndex.has_value())
                    material_index = static_cast<u32>(*prim.materialIndex);
                if (material_index >= gpu_materials.size())
                    material_index = 0;

                const auto vertex_offset = static_cast<u32>(vertices.size());

                vertices.reserve(vertices.size() + positions.size());
                for (size_t i = 0; i < positions.size(); ++i) {
                    Vertex v{};
                    v.position = {positions[i].x, positions[i].y, positions[i].z};

                    auto uv_prior_to_packing = (i < uvs.size()) ? std::array<float, 2>{uvs[i].x, uvs[i].y}
                                                                : std::array<float, 2>{0.0f, 0.0f};
                    v.uvs = glm::packHalf2x16(glm::vec2(uv_prior_to_packing[0], uv_prior_to_packing[1]));

                    const glm::vec3 n = glm::normalize((i < normals.size()) ? normals[i] : glm::vec3(0, 0, 1));
                    v.normal = glm::packSnorm3x10_1x2(glm::vec4(n, 0.0f));

                    const glm::vec4 t4 = (i < tangents.size()) ? tangents[i] : glm::vec4(1, 0, 0, 1);
                    const float handedness = (t4.w < 0.0f) ? -1.0f : 1.0f;
                    v.tangent = glm::packSnorm3x10_1x2(glm::vec4(glm::normalize(glm::vec3(t4)), handedness));
                    v.reserved = 0;
                    vertices.emplace_back(v);
                }

                indices.reserve(indices.size() + local_indices.size());
                for (u32 idx: local_indices)
                    indices.emplace_back(vertex_offset + idx);

                const size_t prim_vertex_count = positions.size();

                std::vector<u32> opt_indices(local_indices.size());
                {
                    std::vector<u32> remap(prim_vertex_count);
                    const size_t unique = meshopt_generateVertexRemap(
                            remap.data(), local_indices.data(), local_indices.size(), vertices.data() + vertex_offset,
                            prim_vertex_count, sizeof(Vertex));

                    meshopt_remapIndexBuffer(opt_indices.data(), local_indices.data(), local_indices.size(),
                                             remap.data());
                    meshopt_optimizeVertexCache(opt_indices.data(), opt_indices.data(), opt_indices.size(), unique);
                    meshopt_optimizeOverdraw(opt_indices.data(), opt_indices.data(), opt_indices.size(),
                                             &vertices[vertex_offset].position[0], prim_vertex_count, sizeof(Vertex),
                                             1.05f);
                }

                constexpr std::array<float, k_lod_count> k_lod_ratios = {1.0f, 0.5f, 0.25f, 0.125f};
                constexpr float k_target_error = 1e-2f;

                Submesh sm{};
                sm.vertex_offset = vertex_offset;
                sm.vertex_count = static_cast<u32>(prim_vertex_count);
                sm.material_index = material_index;

                for (u32 lod = 0; lod < k_lod_count; ++lod) {
                    const u32 global_index_offset = static_cast<u32>(indices.size());

                    if (lod == 0) {
                        for (u32 idx: opt_indices)
                            indices.push_back(vertex_offset + idx);

                        sm.lods[0] = {global_index_offset, static_cast<u32>(opt_indices.size())};
                    } else {
                        const size_t target_count =
                                std::max(3u, static_cast<u32>(opt_indices.size() * k_lod_ratios[lod]));

                        std::vector<u32> simplified(opt_indices.size());
                        float result_error = 0.0f;

                        const size_t simplified_count =
                                meshopt_simplify(simplified.data(), opt_indices.data(), opt_indices.size(),
                                                 &vertices[vertex_offset].position[0], prim_vertex_count,
                                                 sizeof(Vertex), target_count, k_target_error, 0, &result_error);

                        simplified.resize(simplified_count);

                        meshopt_optimizeVertexCache(simplified.data(), simplified.data(), simplified_count,
                                                    prim_vertex_count);

                        for (u32 idx: simplified)
                            indices.push_back(vertex_offset + idx);

                        sm.lods[lod] = {global_index_offset, static_cast<u32>(simplified_count)};

                        info("LOD{}: {}/{} triangles (error {:.4f})", lod, simplified_count / 3, opt_indices.size() / 3,
                             result_error);
                    }
                }

                submeshes.emplace_back(sm);
            }
        }

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
            std::vector<Texture> tex_placeholders(header.texture_count);
            header.texture_table = w.write_pod_array<Texture>(tex_placeholders);
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

        const u64 texture_blob_end = w.size();
        header.texture_blob.offset = texture_blob_begin;
        header.texture_blob.size = texture_blob_end - texture_blob_begin;

        w.patch_pod<FileHeader>(header_offset, header);

        // Final write.
        return write_file_bytes_abs(out_abs_no_normalize, w.data(), src_hash);
    }

} // namespace Tooling
