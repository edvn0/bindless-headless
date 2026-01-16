#include "Compiler.hxx"

#include "Reflection.hxx"

namespace {

    auto load_file_to_string = [](std::filesystem::path const &p) {
        const std::ifstream ifs(p);
        if (!ifs) std::abort();
        std::ostringstream oss;
        oss << ifs.rdbuf();
        return oss.str();
    };
}

CompilerSession::CompilerSession() {
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
            {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr}
        },
        slang::CompilerOptionEntry{
            slang::CompilerOptionName::VulkanUseEntryPointName,
            {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr}
        },
        slang::CompilerOptionEntry{
            slang::CompilerOptionName::Optimization,
            {slang::CompilerOptionValueKind::Int, SLANG_OPTIMIZATION_LEVEL_HIGH, 0, nullptr, nullptr}
        },
        slang::CompilerOptionEntry{
            slang::CompilerOptionName::MatrixLayoutColumn,
            {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr}
        },
        //
    };

    desc.compilerOptionEntries = opts.data();
    desc.compilerOptionEntryCount = static_cast<u32>(opts.size());

    global->createSession(desc, session.writeRef());
}

auto CompilerSession::compile_compute_from_string(std::string_view name, std::string_view path, std::string_view src,
                                                  std::string_view entry) -> std::vector<u32> {
    Slang::ComPtr<slang::IBlob> diagnostics;
    Slang::ComPtr<slang::IModule> module;

    module = session->loadModuleFromSourceString(
        name.data(),
        path.data(),
        src.data(),
        diagnostics.writeRef()
    );

    if (diagnostics) {
        warn("Compiler diagnostic: {}", static_cast<const char *>(diagnostics->getBufferPointer()));
    }

    if (!module) {
        std::abort();
    }

    return compile_compute_module(module, entry);
}

auto CompilerSession::compile_entry_from_string(std::string_view name, std::string_view path, std::string_view src,
                                                std::string_view entry,
                                                ReflectionData *out_reflection) -> std::vector<std::uint32_t> {
    Slang::ComPtr<slang::IBlob> diagnostics;
    Slang::ComPtr<slang::IModule> module;

    module = session->loadModuleFromSourceString(
        name.data(),
        path.data(),
        src.data(),
        diagnostics.writeRef()
    );

    if (diagnostics) {
        warn("Compiler diagnostic: {}", static_cast<const char *>(diagnostics->getBufferPointer()));
    }
    if (!module) {
        std::abort();
    }

    return compile_entry_module(module, entry, out_reflection);
}

auto CompilerSession::compile_entry_from_file(std::string_view path, std::string_view entry,
                                              ReflectionData *out_reflection) -> std::vector<std::uint32_t> {
    auto extract_module_name = [](std::filesystem::path const &p) {
        return p.filename().string();
    };


    std::filesystem::path p{path};
    const auto name = extract_module_name(p);
    const auto src = load_file_to_string(p);

    return compile_entry_from_string(name, path, src, entry, out_reflection);
}

auto CompilerSession::compile_compute_from_file(std::string_view path, std::string_view entry,
                                                ReflectionData *data) -> std::vector<std::uint32_t> {
    return compile_entry_from_file(path, entry, data);
}

auto CompilerSession::compile_compute_module(Slang::ComPtr<slang::IModule> const &module,
                                             std::string_view entry) -> std::vector<u32> {
    Slang::ComPtr<slang::IEntryPoint> ep; {
        Slang::ComPtr<slang::IBlob> diagnostics;
        module->findEntryPointByName(entry.data(), ep.writeRef());
        if (diagnostics) {
            warn("Compiler diagnostic: {}", static_cast<const char *>(diagnostics->getBufferPointer()));
        }
        if (!ep) {
            std::abort();
        }
    }

    std::array<slang::IComponentType *, 2> components = {
        module.get(),
        ep.get()
    };

    Slang::ComPtr<slang::IComponentType> composed; {
        Slang::ComPtr<slang::IBlob> diagnostics;
        auto result = session->createCompositeComponentType(
            components.data(),
            components.size(),
            composed.writeRef(),
            diagnostics.writeRef()
        );
        if (diagnostics) {
            warn("Compiler diagnostic: {}", static_cast<const char *>(diagnostics->getBufferPointer()));
        }
        if (SLANG_FAILED(result)) {
            std::abort();
        }
    }

    Slang::ComPtr<slang::IComponentType> linked; {
        Slang::ComPtr<slang::IBlob> diagnostics;
        auto result = composed->link(linked.writeRef(), diagnostics.writeRef());
        if (diagnostics) {
            warn("Compiler diagnostic: {}", static_cast<const char *>(diagnostics->getBufferPointer()));
        }
        if (SLANG_FAILED(result)) {
            std::abort();
        }
    }

    Slang::ComPtr<slang::IBlob> spirv; {
        Slang::ComPtr<slang::IBlob> diagnostics;
        auto result = linked->getEntryPointCode(0, 0, spirv.writeRef(), diagnostics.writeRef());
        if (diagnostics) {
            warn("Compiler diagnostic: {}", static_cast<const char *>(diagnostics->getBufferPointer()));
        }
        if (SLANG_FAILED(result)) {
            std::abort();
        }
    }

    auto bytes = spirv->getBufferSize();
    std::vector<u32> code(bytes / sizeof(u32));
    std::memcpy(code.data(), spirv->getBufferPointer(), bytes);

    return code;
}

auto CompilerSession::compile_entry_module(Slang::ComPtr<slang::IModule> const &module, std::string_view entry,
                                           ReflectionData *out_reflection) -> std::vector<std::uint32_t> {
    Slang::ComPtr<slang::IEntryPoint> ep; {
        const auto result = module->findEntryPointByName(entry.data(), ep.writeRef());
        if (SLANG_FAILED(result)) {
            error("Could not find entry point with name '{}'", entry);
            std::abort();
        }
    }

    std::array<slang::IComponentType *, 2> components = {
        module.get(),
        ep.get()
    };

    Slang::ComPtr<slang::IComponentType> composed; {
        Slang::ComPtr<slang::IBlob> diagnostics;
        auto result = session->createCompositeComponentType(
            components.data(),
            components.size(),
            composed.writeRef(),
            diagnostics.writeRef()
        );
        if (diagnostics) {
            warn("Compiler diagnostic: {}", static_cast<const char *>(diagnostics->getBufferPointer()));
        }
        if (SLANG_FAILED(result)) {
            std::abort();
        }
    }

    Slang::ComPtr<slang::IComponentType> linked; {
        Slang::ComPtr<slang::IBlob> diagnostics;
        auto result = composed->link(linked.writeRef(), diagnostics.writeRef());
        if (diagnostics) {
            warn("Compiler diagnostic: {}", static_cast<const char *>(diagnostics->getBufferPointer()));
        }
        if (SLANG_FAILED(result)) {
            std::abort();
        }
    }

    if (out_reflection) {
        *out_reflection = reflect_program(linked, /*target_index*/ 0);
    }

    Slang::ComPtr<slang::IBlob> spirv; {
        Slang::ComPtr<slang::IBlob> diagnostics;
        auto result = linked->getEntryPointCode(0, 0, spirv.writeRef(), diagnostics.writeRef());
        if (diagnostics) {
            warn("Compiler diagnostic: {}", static_cast<const char *>(diagnostics->getBufferPointer()));
        }
        if (SLANG_FAILED(result)) {
            std::abort();
        }
    }

    auto bytes = spirv->getBufferSize();
    std::vector<std::uint32_t> code(bytes / sizeof(std::uint32_t));
    std::memcpy(code.data(), spirv->getBufferPointer(), bytes);

    return code;
}

auto CompilerSession::load_shader_file(std::string_view path) -> std::string {
    return load_file_to_string(path);
}
