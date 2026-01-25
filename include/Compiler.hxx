#pragma once

#include "Logger.hxx"
#include "ReflectionData.hxx"
#include "Types.hxx"

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

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

    template<std::size_t N>
    auto compile_from_file(std::string_view path, std::span<const std::string_view, N> entries,
                           std::span<ReflectionData, N> reflection_data) -> std::array<std::vector<u32>, N> {
        std::array<std::vector<u32>, N> spirv{};
        compile_from_file_impl(path, entries, reflection_data, spirv);
        return spirv;
    }

    // Useful if you later want dynamic entry lists
    auto compile_from_file(std::string_view path, std::span<const std::string_view> entries,
                           std::span<ReflectionData> reflection_data) -> std::vector<std::vector<u32>>;

private:
    std::unique_ptr<detail::Impl> impl;

    template<std::size_t N>
    auto compile_from_file_impl(std::string_view path, std::span<const std::string_view, N> entries,
                                std::span<ReflectionData, N> reflection_data,
                                std::array<std::vector<u32>, N> &out_spirv) -> void {
        // Adapter to the dynamic function to avoid duplicating backend logic
        std::vector<std::string_view> dyn_entries(entries.begin(), entries.end());
        std::vector<ReflectionData> dyn_refl(reflection_data.begin(), reflection_data.end());

        auto dyn_spv = compile_from_file(path, dyn_entries, dyn_refl);

        for (std::size_t i = 0; i < N; ++i) {
            out_spirv[i] = std::move(dyn_spv[i]);
            reflection_data[i] = dyn_refl[i];
        }
    }
};
