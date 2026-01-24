#include "Reflection.hxx"

#if !defined(ENGINE_OFFLINE_SHADERS)

#include "Logger.hxx"

namespace {

auto stage_to_string_slang(SlangStage stage) -> std::string_view {
    switch (stage) {
        case SLANG_STAGE_VERTEX: return "vertex";
        case SLANG_STAGE_FRAGMENT: return "fragment";
        case SLANG_STAGE_COMPUTE: return "compute";
        case SLANG_STAGE_GEOMETRY: return "geometry";
        case SLANG_STAGE_HULL: return "hull";
        case SLANG_STAGE_DOMAIN: return "domain";
        case SLANG_STAGE_RAY_GENERATION: return "raygen";
        case SLANG_STAGE_INTERSECTION: return "intersection";
        case SLANG_STAGE_ANY_HIT: return "any_hit";
        case SLANG_STAGE_CLOSEST_HIT: return "closest_hit";
        case SLANG_STAGE_MISS: return "miss";
        case SLANG_STAGE_CALLABLE: return "callable";
        default: return "unknown";
    }
}

auto resource_kind_from_type(slang::TypeLayoutReflection* type_layout) -> std::string {
    if (!type_layout) return {};
    auto* type = type_layout->getType();
    if (!type) return {};

    using K = slang::TypeReflection::Kind;
    switch (type->getKind()) {
        case K::Resource: return "resource";
        case K::SamplerState: return "sampler";
        case K::ConstantBuffer: return "constant_buffer";
        case K::ParameterBlock: return "parameter_block";
        case K::TextureBuffer: return "texture_buffer";
        case K::ShaderStorageBuffer: return "storage_buffer";
        default: break;
    }

    if (auto const* name = type->getName()) {
        return std::string{name};
    }
    return {};
}

auto reflect_parameter_into_entry(ReflectionData::EntryInfo& entry, slang::VariableLayoutReflection* var_layout) -> void {
    if (!var_layout) return;

    auto* type_layout = var_layout->getTypeLayout();
    if (!type_layout) return;

    std::string param_name;
    if (auto const* n = var_layout->getName()) {
        param_name = n;
    }

    // Push constants
    {
        using C = slang::ParameterCategory;
        const int cat_count = var_layout->getCategoryCount();
        for (int ci = 0; ci < cat_count; ++ci) {
            auto const cat = var_layout->getCategoryByIndex(ci);
            if (cat == C::PushConstantBuffer) {
                ReflectionData::PushConstantInfo pc{};
                pc.name = param_name.empty() ? "push_constants" : param_name;
                pc.offset = static_cast<u32>(var_layout->getOffset(cat));
                pc.size = static_cast<u32>(type_layout->getSize(cat));
                entry.push_constants.emplace(pc.name, std::move(pc));
                return;
            }
        }
    }

    // Descriptors: keep your current behavior (set_index hardcoded 0 for now)
    using C = slang::ParameterCategory;

    u32 set_index = 0;
    u32 binding_index = 0;
    bool has_descriptor = false;

    const int cat_count = var_layout->getCategoryCount();
    for (int ci = 0; ci < cat_count; ++ci) {
        auto const cat = var_layout->getCategoryByIndex(ci);
        if (cat == C::DescriptorTableSlot) {
            binding_index = static_cast<u32>(var_layout->getOffset(cat));
            has_descriptor = true;
        }
    }

    if (!has_descriptor) return;

    auto set_key = std::to_string(set_index);
    auto& set = entry.descriptor_sets[set_key];
    if (set.name.empty()) {
        set.name = "set" + std::to_string(set_index);
        set.index = set_index;
    }

    ReflectionData::BindingInfo bi{};
    bi.name = param_name.empty() ? ("binding" + std::to_string(binding_index)) : param_name;
    bi.binding = binding_index;
    bi.set_index = set_index;

    auto* type = type_layout->getType();
    u32 array_size = 1;
    if (type && type->getKind() == slang::TypeReflection::Kind::Array) {
        array_size = static_cast<u32>(type->getElementCount());
    }
    bi.array_size = array_size;

    if (type && type->getName()) {
        bi.type_name = type->getName();
    }

    bi.resource_kind = resource_kind_from_type(type_layout);

    set.bindings.emplace(bi.name, std::move(bi));
}

auto reflect_entry_point(ReflectionData& out, slang::EntryPointReflection* entry_point) -> void {
    if (!entry_point) return;

    auto const stage_sv = stage_to_string_slang(entry_point->getStage());
    std::string stage_key{stage_sv};

    auto& e = out.entries[stage_key];

    if (auto const* n = entry_point->getName()) {
        e.name = n;
    } else {
        e.name = stage_key;
    }
    e.stage_name = stage_key;
    e.stage_flags.insert(stage_key);

    const u32 param_count = static_cast<u32>(entry_point->getParameterCount());
    for (u32 i = 0; i < param_count; ++i) {
        auto* param_layout = entry_point->getParameterByIndex(i);
        reflect_parameter_into_entry(e, param_layout);
    }
}

} // namespace

auto reflect_program(Slang::ComPtr<slang::IComponentType> const& program, int target_index) -> ReflectionData {
    ReflectionData result{};

    if (!program) return result;

    slang::ProgramLayout* layout = program->getLayout(target_index);
    if (!layout) return result;

    const u32 entry_count = static_cast<u32>(layout->getEntryPointCount());
    for (u32 i = 0; i < entry_count; ++i) {
        auto* ep = layout->getEntryPointByIndex(i);
        reflect_entry_point(result, ep);
    }

    return result;
}

#endif
