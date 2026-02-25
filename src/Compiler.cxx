#include "Compiler.hxx"
#include "Reflection.hxx"

#include <array>
#include <fstream>
#include <iterator>
#include <tl/expected.hpp>


#include <slang-com-helper.h>
#include <slang-com-ptr.h>
#include <slang.h>

struct detail::Impl {
    virtual ~Impl() = default;

    virtual auto compile_from_file(std::string_view path, std::span<const std::string_view> entries,
                                   std::span<ReflectionData> reflection_data)
            -> tl::expected<std::vector<std::vector<u32>>, Error> = 0;
};


struct RuntimeSlangCompiler final : detail::Impl {
    Slang::ComPtr<slang::IGlobalSession> global;
    Slang::ComPtr<slang::ISession> session;

    RuntimeSlangCompiler() {
        createGlobalSession(global.writeRef());

        slang::SessionDesc desc{};
        slang::TargetDesc target{};
        target.format = SLANG_SPIRV;
        target.profile = global->findProfile("spirv_1_6_vk");

        desc.targets = &target;
        desc.targetCount = 1;

        std::array<slang::CompilerOptionEntry, 5> opts = {
                slang::CompilerOptionEntry{
                        .name = slang::CompilerOptionName::EmitSpirvDirectly,
                        .value =
                                {
                                        .kind = slang::CompilerOptionValueKind::Int,
                                        .intValue0 = 1,
                                        .intValue1 = 0,
                                        .stringValue0 = nullptr,
                                        .stringValue1 = nullptr,
                                },
                },
                slang::CompilerOptionEntry{
                        .name = slang::CompilerOptionName::VulkanUseEntryPointName,
                        .value =
                                {
                                        .kind = slang::CompilerOptionValueKind::Int,
                                        .intValue0 = 1,
                                        .intValue1 = 0,
                                        .stringValue0 = nullptr,
                                        .stringValue1 = nullptr,
                                },
                },
                slang::CompilerOptionEntry{
                        .name = slang::CompilerOptionName::Optimization,
                        .value =
                                {
                                        .kind = slang::CompilerOptionValueKind::Int,
                                        .intValue0 = SLANG_OPTIMIZATION_LEVEL_HIGH,
                                        .intValue1 = 0,
                                        .stringValue0 = nullptr,
                                        .stringValue1 = nullptr,
                                },
                },
                slang::CompilerOptionEntry{
                        .name = slang::CompilerOptionName::MatrixLayoutColumn,
                        .value =
                                {
                                        .kind = slang::CompilerOptionValueKind::Int,
                                        .intValue0 = 1,
                                        .intValue1 = 0,
                                        .stringValue0 = nullptr,
                                        .stringValue1 = nullptr,
                                },
                },
                slang::CompilerOptionEntry{
                        .name = slang::CompilerOptionName::DebugInformation,
                        .value =
                                {
                                        .kind = slang::CompilerOptionValueKind::Int,
                                        .intValue0 = SLANG_DEBUG_INFO_LEVEL_MAXIMAL,
                                        .intValue1 = 0,
                                        .stringValue0 = nullptr,
                                        .stringValue1 = nullptr,
                                },
                },
        };

        desc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;
        desc.compilerOptionEntries = opts.data();
        desc.compilerOptionEntryCount = static_cast<u32>(opts.size());

        global->createSession(desc, session.writeRef());
    }

    auto load_file_to_string(std::filesystem::path const &p) -> std::string {
        std::ifstream ifs(p);
        if (!ifs) {
            error("Could not open file {}", p.string());
            return {};
        }
        std::ostringstream oss;
        oss << ifs.rdbuf();
        return oss.str();
    }

    auto compile_entry_module(Slang::ComPtr<slang::IModule> const &slang_module, std::string_view entry,
                              ReflectionData *out_reflection) -> std::vector<u32> {
        Slang::ComPtr<slang::IEntryPoint> ep;
        {
            const auto r = slang_module->findEntryPointByName(entry.data(), ep.writeRef());
            if (SLANG_FAILED(r) || !ep) {
                error("Could not find entry point '{}'", entry);
                std::abort();
            }
        }

        std::array<slang::IComponentType *, 2> components = {slang_module.get(), ep.get()};

        Slang::ComPtr<slang::IComponentType> composed;
        {
            Slang::ComPtr<slang::IBlob> diagnostics;
            const auto r = session->createCompositeComponentType(components.data(), components.size(),
                                                                 composed.writeRef(), diagnostics.writeRef());
            if (diagnostics) {
                warn("Compiler diagnostic: {}", static_cast<const char *>(diagnostics->getBufferPointer()));
            }
            if (SLANG_FAILED(r)) {
                std::abort();
            }
        }

        Slang::ComPtr<slang::IComponentType> linked;
        {
            Slang::ComPtr<slang::IBlob> diagnostics;
            const auto r = composed->link(linked.writeRef(), diagnostics.writeRef());
            if (diagnostics) {
                warn("Compiler diagnostic: {}", static_cast<const char *>(diagnostics->getBufferPointer()));
            }
            if (SLANG_FAILED(r)) {
                std::abort();
            }
        }

        if (out_reflection) {
            *out_reflection = reflect_program(linked, /*target_index*/ 0);
        }

        Slang::ComPtr<slang::IBlob> spirv;
        {
            Slang::ComPtr<slang::IBlob> diagnostics;
            const auto r = linked->getEntryPointCode(0, 0, spirv.writeRef(), diagnostics.writeRef());
            if (diagnostics) {
                warn("Compiler diagnostic: {}", static_cast<const char *>(diagnostics->getBufferPointer()));
            }
            if (SLANG_FAILED(r)) {
                std::abort();
            }
        }

        std::vector<u32> code(spirv->getBufferSize() / sizeof(u32));
        std::memcpy(code.data(), spirv->getBufferPointer(), spirv->getBufferSize());
        return code;
    }

    auto compile_from_file(std::string_view path, std::span<const std::string_view> entries,
                           std::span<ReflectionData> reflection_data)
            -> tl::expected<std::vector<std::vector<u32>>, Error> override {
        std::filesystem::path p{path};

        // Make both name AND path unique to bypass Slang's internal caching
        const auto timestamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        const auto name = p.filename().string() + "_" + timestamp;
        const auto unique_path = std::string(path) + "?" + timestamp; // Add query-string-like suffix

        const auto src = load_file_to_string(p);
        std::vector<std::vector<u32>> result;
        result.resize(entries.size());

        if (src.empty()) {
            error("Shader source empty: {}", p.string());
            for (std::size_t i = 0; i < std::min(entries.size(), reflection_data.size()); ++i) {
                reflection_data[i] = ReflectionData{};
            }
            return tl::make_unexpected(Error::make_error(Error::Type::FileNotFoundError, "Empty file at {}", path));
        }

        Slang::ComPtr<slang::IBlob> diagnostics;
        Slang::ComPtr<slang::IModule> slang_module_from_session;
        slang_module_from_session = session->loadModuleFromSourceString(name.c_str(), unique_path.c_str(), src.c_str(),
                                                                        diagnostics.writeRef());

        if (diagnostics) {
            warn("Compiler diagnostic: {}", static_cast<const char *>(diagnostics->getBufferPointer()));
        }
        if (!slang_module_from_session) {
            return tl::make_unexpected(
                    Error::make_error(Error::Type::ShaderCompileError, "Failed to compiler shader."));
        }

        for (std::size_t i = 0; i < entries.size(); ++i) {
            ReflectionData *out_refl = nullptr;
            if (i < reflection_data.size()) {
                out_refl = &reflection_data[i];
            }
            result[i] = compile_entry_module(slang_module_from_session, entries[i], out_refl);
        }

        return result;
    }
};

Compiler::Compiler() {
    impl = std::make_unique<RuntimeSlangCompiler>();
}

Compiler::~Compiler() = default;

Compiler::Compiler(Compiler &&) noexcept = default;
auto Compiler::operator=(Compiler &&) noexcept -> Compiler & = default;

auto Compiler::compile_from_file(std::string_view path, std::span<const std::string_view> entries,
                                 std::span<ReflectionData> reflection_data)
        -> tl::expected<std::vector<std::vector<u32>>, Error> {

    std::filesystem::path p{path};
    if (!std::filesystem::exists(p)) {
        return tl::make_unexpected(Error::make_error(Error::Type::FileNotFoundError, std::string(path)));
    }

    auto current_write_time = std::filesystem::last_write_time(p);

    if (auto it = disk_cache.find(path); it != disk_cache.end()) {
        if (it->second.last_write_time == current_write_time) {
            for (size_t i = 0; i < std::min(reflection_data.size(), it->second.reflection_results.size()); ++i) {
                reflection_data[i] = it->second.reflection_results[i];
            }
            return it->second.spirv_results;
        } else {
            disk_cache.erase(it);
        }
    }

    auto result = impl->compile_from_file(path, entries, reflection_data);

    if (result) {
        disk_cache[std::string(path)] =
                CacheEntry{.last_write_time = current_write_time,
                           .spirv_results = *result,
                           .reflection_results = {reflection_data.begin(), reflection_data.end()}};
    }

    return result;
}
