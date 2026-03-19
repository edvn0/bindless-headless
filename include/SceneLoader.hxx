// SceneLoader.hxx
#pragma once

#include <filesystem>
#include <tl/expected.hpp>

#include "Error.hxx"
#include "Material.hxx"
#include "Numeric.hxx"
#include "Types.hxx"

#include <array>
#include <bit>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Tooling {

    static constexpr u32 k_version = 2;
    static constexpr u32 k_lod_count = 4;

    struct LodRange {
        u32 index_offset = 0;
        u32 index_count = 0; // 0 = this LOD not present
    };
    static_assert(sizeof(LodRange) == 8);

    struct Submesh {
        u32 vertex_offset = 0;
        u32 vertex_count = 0;
        u32 material_index = 0;
        u32 reserved0 = 0;
        std::array<LodRange, k_lod_count> lods{};
    };
    static_assert(sizeof(Submesh) == 48);
    static_assert(std::is_trivially_copyable_v<Submesh>);

    struct BlobRange {
        u64 offset = 0; // from file start
        u64 size = 0; // bytes
    };

    inline auto align_up_u64(u64 v, u64 a) -> u64 { return (v + (a - 1)) & ~(a - 1); }

    class BinaryWriter {
    public:
        [[nodiscard]] auto size() const -> u64 { return m_bytes.size(); }
        [[nodiscard]] auto data() const -> std::span<const std::byte> { return m_bytes; }
        [[nodiscard]] auto data() -> std::span<std::byte> { return m_bytes; }

        auto align(const u64 alignment) -> void {
            const auto new_size = align_up_u64(size(), alignment);
            m_bytes.resize(new_size, std::byte{0});
        }

        template<class T>
        auto write_pod(const T &v) -> u64 {
            static_assert(std::is_trivially_copyable_v<T>);
            const u64 off = size();
            const auto *p = std::bit_cast<const std::byte *>(&v);
            m_bytes.insert(m_bytes.end(), p, p + sizeof(T));
            return off;
        }

        template<class T>
        auto write_pod_array(std::span<const T> arr) -> BlobRange {
            static_assert(std::is_trivially_copyable_v<T>);
            const u64 off = size();
            const auto bytes = std::as_bytes(arr);
            m_bytes.insert(m_bytes.end(), bytes.begin(), bytes.end());
            return BlobRange{off, static_cast<u64>(bytes.size())};
        }

        auto write_bytes(std::span<const std::byte> bytes) -> BlobRange {
            const u64 off = size();
            m_bytes.insert(m_bytes.end(), bytes.begin(), bytes.end());
            return BlobRange{off, bytes.size()};
        }

        template<class T>
        auto patch_pod(u64 file_offset, const T &v) -> void {
            static_assert(std::is_trivially_copyable_v<T>);
            std::memcpy(m_bytes.data() + file_offset, &v, sizeof(T));
        }

    private:
        std::vector<std::byte> m_bytes;
    };

    class StringTable {
    public:
        auto add(std::string_view s) -> u32 {
            if (s.empty())
                return 0;

            if (auto it = m_offsets.find(s); it != m_offsets.end())
                return it->second;

            const auto off = static_cast<u32>(m_blob.size());
            m_blob.insert(m_blob.end(), s.begin(), s.end());
            m_blob.push_back('\0');
            m_offsets.try_emplace(std::string{s}, off);
            return off;
        }

        auto blob() const -> std::span<const std::byte> {
            return std::as_bytes(std::span<const char>(m_blob.data(), m_blob.size()));
        }

    private:
        StringMap<u32> m_offsets;
        std::vector<char> m_blob;
    };

    static constexpr u32 k_magic = 0x31534E43; // 'CNS1'

    struct FileHeader {
        u32 magic = k_magic;
        u32 version = k_version;

        u32 flags = 0;
        u32 reserved0 = 0;

        u32 submesh_count = 0;
        u32 vertex_count = 0;
        u32 index_count = 0;
        u32 material_count = 0;
        u32 texture_count = 0;

        u32 reserved1 = 0;
        u64 content_hash = 0; // optional

        BlobRange submesh_table;
        BlobRange vertex_blob;
        BlobRange index_blob;
        BlobRange material_table;
        BlobRange texture_table;
        BlobRange string_blob;
        BlobRange texture_blob; // concatenated KTX2 files
    };

    struct Vertex {
        std::array<float, 3> position{};
        u32 uvs{}; // packed half2 (x=u, y=v)
        u32 normal{}; // packed 10_10_10_2
        u32 tangent{}; // packed 10_10_10_2 (xyz + sign in w/2 bits)
        u32 reserved{}; // note: keeps sizeof(Vertex)==28
    };
    static_assert(sizeof(Vertex) == 28);
    static_assert(std::is_trivially_copyable_v<Vertex>);
    static_assert(alignof(Vertex) == 4);

    struct GPUMaterial {
        u32 albedo_map = 0xFFFFFFFFu;
        std::array<float, 4> albedo_factor{1, 1, 1, 1};

        u32 normal_map = 0xFFFFFFFFu;
        u32 roughness_map = 0xFFFFFFFFu;
        float roughness_factor = 1.0f;

        u32 metallic_map = 0xFFFFFFFFu;
        float metallic_factor = 1.0f;

        u32 occlusion_map = 0xFFFFFFFFu;
        u32 emissive_map = 0xFFFFFFFFu;
        std::array<float, 3> emissive_factor{0, 0, 0};

        MaterialFlags flags = MaterialFlags::None;
        u32 reserved0 = 0;
    };

    struct Texture {
        u32 original_path_str = 0;
        u32 name_str = 0;
        u32 reserved0 = 0;
        u32 reserved1 = 0;

        u64 ktx2_offset = 0;
        u64 ktx2_size = 0;
    };
    static_assert(sizeof(Texture) == 32);

    class SceneLoader {
    public:
        explicit SceneLoader(const std::filesystem::path &meshes_root = "assets/meshes") : m_meshes_root(meshes_root) {}

        auto convert_gltf(const std::filesystem::path &scene_path, const std::filesystem::path &output_path)
                -> tl::expected<void, Error>;

        [[nodiscard]] auto meshes_root() const -> const std::filesystem::path & { return m_meshes_root; }

    private:
        std::filesystem::path m_meshes_root;

        static auto resolve_under(const std::filesystem::path &root, const std::filesystem::path &p)
                -> std::filesystem::path {
            if (p.is_absolute())
                return p;
            return root / p;
        }
    };

} // namespace Tooling
