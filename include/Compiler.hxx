#pragma once

#include "Logger.hxx"
#include "Forward.hxx"
#include "Types.hxx"

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

private:
    auto compile_compute_module(
        Slang::ComPtr<slang::IModule> const &module,
        std::string_view entry) -> std::vector<u32>;

    auto compile_entry_module(
        Slang::ComPtr<slang::IModule> const &module,
        std::string_view entry,
        ReflectionData *out_reflection) -> std::vector<std::uint32_t>;
};
