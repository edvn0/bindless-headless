#pragma once

#include "Error.hxx"
#include "Numeric.hxx"

#include <filesystem>
#include <span>
#include <tl/expected.hpp>
#include <vector>

namespace detail {
    constexpr auto file_end = ".cscene";
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

auto scene_decompress(std::span<const std::byte>) -> tl::expected<std::vector<std::byte>, Error>;
auto scene_compress_to_memory(std::span<const std::byte> data, u64 src_hash)
        -> tl::expected<std::vector<std::byte>, Error>;
auto scene_compress(const std::filesystem::path &, std::span<std::byte>, u64) -> tl::expected<void, Error>;
