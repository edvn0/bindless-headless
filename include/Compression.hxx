#pragma once

#include "Error.hxx"
#include "Numeric.hxx"

#include <filesystem>
#include <span>
#include <tl/expected.hpp>
#include <vector>

namespace detail {
    constexpr auto file_end = ".scene.bz2";
}

inline auto make_error(std::string msg) -> Error {
    return Error{.type = Error::Type::SceneLoaderError, .message = std::move(msg)};
}

constexpr u64 k_align = 16;
constexpr u64 k_prefix_magic = 0x454E4543534E5331ULL; // 'SNS1CENE'

inline auto normalize_scene_out_path(std::filesystem::path out_path) -> std::filesystem::path {
    if (out_path.has_extension()) {
        out_path.replace_extension(detail::file_end);
    } else {
        out_path += detail::file_end;
    }
    return out_path;
}

#if defined(SCENE_COMPRESSION_BZIP2)

auto bzip2_compress(const std::filesystem::path &out_path_no_normalize, std::span<std::byte> bytes, u64 src_hash)
        -> tl::expected<void, Error>;

auto bzip2_decompress(std::span<const std::byte> src) -> tl::expected<std::vector<std::byte>, Error>;

inline auto scene_compress(const std::filesystem::path &out_path, std::span<std::byte> bytes, u64 src_hash) {
    return bzip2_compress(out_path, bytes, src_hash);
}
inline auto scene_decompress(std::span<const std::byte> src) { return bzip2_decompress(src); }

#elif defined(SCENE_COMPRESSION_ZSTD)

auto zstd_compress(const std::filesystem::path &out_path_no_normalize, std::span<std::byte> bytes, u64 src_hash)
        -> tl::expected<void, Error>;

auto zstd_decompress(std::span<const std::byte> src) -> tl::expected<std::vector<std::byte>, Error>;

inline auto scene_compress(const std::filesystem::path &out_path, std::span<std::byte> bytes, u64 src_hash) {
    return zstd_compress(out_path, bytes, src_hash);
}
inline auto scene_decompress(std::span<const std::byte> src) { return zstd_decompress(src); }

#elif defined(SCENE_COMPRESSION_LZ4)

auto lz4_compress(const std::filesystem::path &out_path_no_normalize, std::span<std::byte> bytes, u64 src_hash)
        -> tl::expected<void, Error>;

auto lz4_decompress(std::span<const std::byte> src) -> tl::expected<std::vector<std::byte>, Error>;

inline auto scene_compress(const std::filesystem::path &out_path, std::span<std::byte> bytes, u64 src_hash) {
    return lz4_compress(out_path, bytes, src_hash);
}
inline auto scene_decompress(std::span<const std::byte> src) { return lz4_decompress(src); }

#else
#error "No scene compression backend defined. Set SCENE_COMPRESSION in CMake."
#endif
