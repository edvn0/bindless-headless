#pragma once

#include <filesystem>
#include <tl/expected.hpp>

#include "Error.hxx"
#include "Numeric.hxx"

#include <array>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace Tooling {

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
        struct TransparentStringHash {
            using is_transparent = void;

            auto operator()(std::string_view s) const noexcept -> usize { return std::hash<std::string_view>{}(s); }

            auto operator()(const std::string &s) const noexcept -> usize { return std::hash<std::string_view>{}(s); }

            auto operator()(const char *s) const noexcept -> usize { return std::hash<std::string_view>{}(s); }
        };

        std::unordered_map<std::string, u32, TransparentStringHash, std::equal_to<>> m_offsets;
        std::vector<char> m_blob; // null-terminated strings
    };

    static constexpr u32 k_magic = 0x31534E43; // 'CNS1' (pick any)
    static constexpr u32 k_version = 1;


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

    // Keep your 10_10_10_2 packing; recommend tangent.w = handedness.
    struct Vertex {
        std::array<float, 3> position{};
        std::array<float, 2> uv0{};
        u32 normal; // packed 10_10_10_2
        u32 tangent; // packed 10_10_10_2 (xyz + sign in w or last bits)
        u32 reserved; // pad to 32 bytes
    };
    static_assert(sizeof(Vertex) == 32);
    static_assert(std::is_trivially_copyable_v<Vertex>);
    static_assert(alignof(Vertex) == 4);

    struct Submesh {
        u32 vertex_offset = 0;
        u32 vertex_count = 0;
        u32 index_offset = 0;
        u32 index_count = 0;
        u32 material_index = 0;
        u32 reserved0 = 0;
        u64 reserved1 = 0;
    };
    static_assert(sizeof(Submesh) == 32);
    static_assert(std::is_trivially_copyable_v<Submesh>);
    static_assert(alignof(Submesh) == 8);

    // This is the runtime GPU material payload (your struct, but POD-safe).
    // IMPORTANT: GLM types are not ABI-stable on disk -> use float arrays.
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

        u32 flags = 0;
        u32 reserved0 = 0;
    };

    // Texture entry points to one KTX2 byte-range in texture_blob.
    // original_path/name are optional debug strings; runtime can ignore.
    struct Texture {
        u32 original_path_str = 0; // offset into string_blob (0 means none)
        u32 name_str = 0; // optional
        u32 reserved0 = 0;
        u32 reserved1 = 0;

        u64 ktx2_offset = 0; // file offset to KTX2 bytes (NOT relative to texture_blob)
        u64 ktx2_size = 0;
    };
    static_assert(sizeof(Texture) == 32);

    class SceneLoader {
    public:
        /**
         * @brief Converts a GLTF file to a serialised scene format. Performs ktx mipmapping and texture packing as part
         * of the conversion.
         * @param scene_path The path to the GLTF file.
         * @param output_path The path to the output file.
         * @return tl::expected<void, Error> indicating success or failure.
         */
        auto convert_gltf(const std::filesystem::path &scene_path, const std::filesystem::path &output_path)
                -> tl::expected<void, Error>;
    };
} // namespace Tooling
