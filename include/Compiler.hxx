#pragma once

#include "Logger.hxx"
#include "Forward.hxx"
#include "Types.hxx"

#include <vector>

#include <slang/slang.h>
#include <slang/slang-com-ptr.h>
#include <slang/slang-com-helper.h>

struct CompilerSession {
    Slang::ComPtr<slang::IGlobalSession> global;
    Slang::ComPtr<slang::ISession> session;

    CompilerSession();

    auto compile_compute_from_string(
        std::string_view name,
        std::string_view path,
        std::string_view src,
        std::string_view entry = "main") -> std::vector<u32>;

    auto compile_entry_from_string(
        std::string_view name,
        std::string_view path,
        std::string_view src,
        std::string_view entry,
        ReflectionData *out_reflection = nullptr) -> std::vector<std::uint32_t>;

    auto compile_entry_from_file(
        std::string_view path,
        std::string_view entry,
        ReflectionData *out_reflection = nullptr) -> std::vector<std::uint32_t>;

    // previous compute-only helpers can forward to this if you wish
    auto compile_compute_from_file(
        std::string_view path,
        std::string_view entry, ReflectionData *data = nullptr) -> std::vector<std::uint32_t>;

    template<std::size_t N>
    auto compile_from_file(std::string_view path, const std::span<const std::string_view, N> entries, const std::span<ReflectionData, N> reflection_data) {
        std::array<std::vector<std::uint32_t>, N> spirv_data{};
        const auto shader_source = load_shader_file(path);

        if (shader_source.empty()) return spirv_data;

        for (std::size_t i = 0; i < N; ++i) {
            std::filesystem::path entry_path(path);
            spirv_data[i] = compile_entry_from_string(entry_path.filename().string(), path, shader_source, entries[i], &reflection_data[i]);
        }
        return spirv_data;
    }

private:
    auto compile_compute_module(
        Slang::ComPtr<slang::IModule> const &module,
        std::string_view entry) -> std::vector<u32>;

    auto compile_entry_module(
        Slang::ComPtr<slang::IModule> const &module,
        std::string_view entry,
        ReflectionData *out_reflection) -> std::vector<std::uint32_t>;

    auto load_shader_file(std::string_view) -> std::string;
};
