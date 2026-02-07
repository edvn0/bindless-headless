#pragma once

#include "Logger.hxx"
#include "ReflectionData.hxx"
#include "Types.hxx"

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <tl/expected.hpp>
#include <unordered_map>
#include <vector>

struct CacheEntry {
    std::filesystem::file_time_type last_write_time;
    std::vector<std::vector<u32>> spirv_results;
    std::vector<ReflectionData> reflection_results;
};

namespace detail {
    struct Impl;
}

class Compiler {
public:
    Compiler();
    ~Compiler();

    Compiler(Compiler const &) = delete;
    auto operator=(Compiler const &) -> Compiler & = delete;
    Compiler(Compiler &&) noexcept;
    auto operator=(Compiler &&) noexcept -> Compiler &;

    // Fixed-size helper for fixed entry points
    template<std::size_t N>
    auto compile_from_file(std::string_view path, std::span<const std::string_view, N> entries,
                           std::span<ReflectionData, N> reflection_data)
            -> tl::expected<std::array<std::vector<u32>, N>, Error> {

        std::vector<std::string_view> dyn_entries(entries.begin(), entries.end());
        std::vector<ReflectionData> dyn_refl(N);

        auto result = compile_from_file(path, dyn_entries, dyn_refl);
        if (!result)
            return tl::make_unexpected(result.error());

        std::array<std::vector<u32>, N> out_spirv;
        for (std::size_t i = 0; i < N; ++i) {
            out_spirv[i] = std::move((*result)[i]);
            reflection_data[i] = std::move(dyn_refl[i]);
        }
        return out_spirv;
    }

    // Base dynamic method
    auto compile_from_file(std::string_view path, std::span<const std::string_view> entries,
                           std::span<ReflectionData> reflection_data)
            -> tl::expected<std::vector<std::vector<u32>>, Error>;

private:
    std::unique_ptr<detail::Impl> impl;
    std::unordered_map<std::string, CacheEntry, string_hash, string_eq> disk_cache;
};
