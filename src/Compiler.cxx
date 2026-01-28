#include "Compiler.hxx"

#include <fstream>
#include <iterator>

#if !defined(ENGINE_OFFLINE_SHADERS)
#include <slang-com-helper.h>
#include <slang-com-ptr.h>
#include <slang.h>
#endif

#if defined(ENGINE_OFFLINE_SHADERS)
namespace {

    auto read_file_bytes(std::filesystem::path const &p) -> std::vector<std::byte> {
        std::ifstream ifs(p, std::ios::binary);
        if (!ifs) {
            error("Could not open file {}", p.string());
            return {};
        }

        ifs.seekg(0, std::ios::end);
        const auto size = static_cast<std::size_t>(ifs.tellg());
        ifs.seekg(0, std::ios::beg);

        std::vector<std::byte> data(size);
        if (!data.empty()) {
            ifs.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size()));
        }
        return data;
    }

    auto bytes_to_u32_words(std::vector<std::byte> const &bytes, std::filesystem::path const &p) -> std::vector<u32> {
        if ((bytes.size() % sizeof(u32)) != 0) {
            error("SPIR-V file size not multiple of 4: {} ({} bytes)", p.string(), bytes.size());
            return {};
        }

        std::vector<u32> words(bytes.size() / sizeof(u32));
        if (!words.empty()) {
            std::memcpy(words.data(), bytes.data(), bytes.size());
        }
        return words;
    }

    auto module_stem(std::string_view slang_path) -> std::string {
        std::filesystem::path p{slang_path};
        return p.filename().replace_extension("").string();
    }

    auto offline_spv_path(std::string_view slang_path, std::string_view entry) -> std::filesystem::path {
        // Must match CMake output naming: <module>__<entry>.spv
        // And must match output folder copied next to the exe: shaders_spv/
        return std::filesystem::path("shaders_spv") / (module_stem(slang_path) + "__" + std::string(entry) + ".spv");
    }

} // namespace
#endif

struct detail::Impl {
    virtual ~Impl() = default;

    virtual auto compile_from_file(std::string_view path, std::span<const std::string_view> entries,
                                   std::span<ReflectionData> reflection_data) -> std::vector<std::vector<u32>> = 0;
};

#if defined(ENGINE_OFFLINE_SHADERS)

struct OfflineSpvCompiler final : detail::Impl {
    auto compile_from_file(std::string_view path, std::span<const std::string_view> entries,
                           std::span<ReflectionData> reflection_data) -> std::vector<std::vector<u32>> override {
        std::vector<std::vector<u32>> result;
        result.resize(entries.size());

        // No reflection yet (later: SPIRV-Reflect)
        for (std::size_t i = 0; i < std::min(entries.size(), reflection_data.size()); ++i) {
            reflection_data[i] = ReflectionData{};
        }

        for (std::size_t i = 0; i < entries.size(); ++i) {
            auto spv = offline_spv_path(path, entries[i]);
            auto bytes = read_file_bytes(spv);
            if (bytes.empty()) {
                error("Missing/empty SPIR-V for {} entry {} (expected {})", path, entries[i], spv.string());
                result[i] = {};
                continue;
            }
            result[i] = bytes_to_u32_words(bytes, spv);
        }

        return result;
    }
};

#else

// Your existing reflection helper
#include "Reflection.hxx"

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

        std::array opts = {
                slang::CompilerOptionEntry{
                        slang::CompilerOptionName::EmitSpirvDirectly,
                        {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr},
                },
                slang::CompilerOptionEntry{
                        slang::CompilerOptionName::VulkanUseEntryPointName,
                        {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr},
                },
                slang::CompilerOptionEntry{
                        slang::CompilerOptionName::Optimization,
                        {slang::CompilerOptionValueKind::Int, SLANG_OPTIMIZATION_LEVEL_HIGH, 0, nullptr, nullptr},
                },
                slang::CompilerOptionEntry{
                        slang::CompilerOptionName::MatrixLayoutColumn,
                        {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr},
                },
                slang::CompilerOptionEntry{
                        slang::CompilerOptionName::DebugInformation,
                        {slang::CompilerOptionValueKind::Int, SLANG_DEBUG_INFO_LEVEL_MAXIMAL, 0, nullptr, nullptr},
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
                           std::span<ReflectionData> reflection_data) -> std::vector<std::vector<u32>> override {
        std::filesystem::path p{path};
        const auto name = p.filename().string();
        const auto src = load_file_to_string(p);

        std::vector<std::vector<u32>> result;
        result.resize(entries.size());

        if (src.empty()) {
            error("Shader source empty: {}", p.string());
            for (std::size_t i = 0; i < std::min(entries.size(), reflection_data.size()); ++i) {
                reflection_data[i] = ReflectionData{};
            }
            return result;
        }

        Slang::ComPtr<slang::IBlob> diagnostics;
        Slang::ComPtr<slang::IModule> slang_module_from_session;
        slang_module_from_session =
                session->loadModuleFromSourceString(name.c_str(), path.data(), src.c_str(), diagnostics.writeRef());

        if (diagnostics) {
            warn("Compiler diagnostic: {}", static_cast<const char *>(diagnostics->getBufferPointer()));
        }
        if (!slang_module_from_session) {
            std::abort();
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

#endif

Compiler::Compiler() {
#if defined(ENGINE_OFFLINE_SHADERS)
    impl = std::make_unique<OfflineSpvCompiler>();
#else
    impl = std::make_unique<RuntimeSlangCompiler>();
#endif
}

Compiler::~Compiler() = default;

Compiler::Compiler(Compiler &&) noexcept = default;
auto Compiler::operator=(Compiler &&) noexcept -> Compiler & = default;

auto Compiler::compile_from_file(std::string_view path, std::span<const std::string_view> entries,
                                 std::span<ReflectionData> reflection_data) -> std::vector<std::vector<u32>> {
    if (reflection_data.size() < entries.size()) {
        // keep you honest: you rely on parallel arrays everywhere
        warn("Reflection span smaller than entries span ({} < {})", reflection_data.size(), entries.size());
    }
    return impl->compile_from_file(path, entries, reflection_data);
}
