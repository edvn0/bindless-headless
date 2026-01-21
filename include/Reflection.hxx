#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Types.hxx"

#include <slang/slang-com-helper.h>
#include <slang/slang-com-ptr.h>
#include <slang/slang.h>


// Heterogeneous string hashing / equality
struct string_hash {
    using is_transparent = void;

    auto operator()(std::string_view v) const noexcept -> std::size_t { return std::hash<std::string_view>{}(v); }

    auto operator()(std::string const &s) const noexcept -> std::size_t { return (*this)(std::string_view{s}); }

    auto operator()(char const *s) const noexcept -> std::size_t { return (*this)(std::string_view{s}); }
};

struct string_eq {
    using is_transparent = void;

    auto operator()(std::string_view a, std::string_view b) const noexcept -> bool { return a == b; }
};

struct ReflectionData {
    struct PushConstantInfo {
        std::string name;
        std::uint32_t offset{};
        std::uint32_t size{};

        std::unordered_map<std::string, std::uint32_t, string_hash, string_eq> fields;
    };

    struct BindingInfo {
        std::string name;
        std::uint32_t binding{};
        std::uint32_t array_size{1};
        std::string type_name;
        std::string resource_kind;
        std::uint32_t set_index{};
    };

    struct SetInfo {
        std::string name;
        std::uint32_t index{};

        std::unordered_map<std::string, BindingInfo, string_hash, string_eq> bindings;
    };

    struct EntryInfo {
        // logical key for this entry: we will key by stage string ("compute",
        // "vertex", "fragment", ...) but keep both here:
        std::string name; // entry-point function name in Slang
        std::string stage_name; // "compute", "vertex", ...

        std::unordered_map<std::string, PushConstantInfo, string_hash, string_eq> push_constants;

        std::unordered_map<std::string, SetInfo, string_hash, string_eq> descriptor_sets;

        std::unordered_set<std::string, string_hash, string_eq> stage_flags;

        std::unordered_map<std::string, std::string, string_hash, string_eq> attributes;
    };

    std::unordered_map<std::string, EntryInfo, string_hash, string_eq> entries;

    std::unordered_map<std::string, std::string, string_hash, string_eq> defines;

    std::unordered_map<std::string, std::string, string_hash, string_eq> types;

    std::unordered_map<std::string, std::string, string_hash, string_eq> capabilities;

    std::unordered_map<std::string, std::string, string_hash, string_eq> misc;
};


// Map slang::Stage to a stable string key
inline auto stage_to_string(SlangStage stage) -> std::string_view {
    using S = SlangStage;
    switch (stage) {
        case SLANG_STAGE_VERTEX:
            return "vertex";
        case SLANG_STAGE_FRAGMENT:
            return "fragment";
        case SLANG_STAGE_COMPUTE:
            return "compute";
        case SLANG_STAGE_GEOMETRY:
            return "geometry";
        case SLANG_STAGE_HULL:
            return "hull";
        case SLANG_STAGE_DOMAIN:
            return "domain";
        case SLANG_STAGE_RAY_GENERATION:
            return "raygen";
        case SLANG_STAGE_INTERSECTION:
            return "intersection";
        case SLANG_STAGE_ANY_HIT:
            return "any_hit";
        case SLANG_STAGE_CLOSEST_HIT:
            return "closest_hit";
        case SLANG_STAGE_MISS:
            return "miss";
        case SLANG_STAGE_CALLABLE:
            return "callable";
        default:
            return "unknown";
    }
}

// Very simple resource-kind string from TypeReflection
inline auto resource_kind_from_type(slang::TypeLayoutReflection *type_layout) -> std::string {
    if (!type_layout) {
        return {};
    }

    auto *type = type_layout->getType();
    if (!type) {
        return {};
    }

    using K = slang::TypeReflection::Kind;
    switch (type->getKind()) {
        case K::Resource:
            return "resource";
        case K::SamplerState:
            return "sampler";
        case K::ConstantBuffer:
            return "constant_buffer";
        case K::ParameterBlock:
            return "parameter_block";
        case K::TextureBuffer:
            return "texture_buffer";
        case K::ShaderStorageBuffer:
            return "storage_buffer";
        default:
            break;
    }

    if (auto const *name = type->getName()) {
        return std::string{name};
    }
    return {};
}

// Reflect a single parameter layout into entry_info
inline auto reflect_parameter_into_entry(ReflectionData::EntryInfo &entry, slang::VariableLayoutReflection *var_layout)
        -> void {
    if (!var_layout) {
        return;
    }

    auto *type_layout = var_layout->getTypeLayout();
    if (!type_layout) {
        return;
    }

    std::string param_name;
    if (auto const *n = var_layout->getName()) {
        param_name = n;
    }

    // 1) Push constants (SPIR-V-specific category)
    {
        using C = slang::ParameterCategory;
        const int cat_count = var_layout->getCategoryCount();
        for (int ci = 0; ci < cat_count; ++ci) {
            auto const cat = var_layout->getCategoryByIndex(ci);
            if (cat == C::PushConstantBuffer) {
                ReflectionData::PushConstantInfo pc{};
                pc.name = param_name.empty() ? "push_constants" : param_name;
                pc.offset = static_cast<std::uint32_t>(var_layout->getOffset(cat));
                pc.size = static_cast<std::uint32_t>(type_layout->getSize(cat));
                entry.push_constants.emplace(pc.name, std::move(pc));
                return;
            }
        }
    }

    // 2) Descriptors (sets + bindings, Vulkan-style)
    using C = slang::ParameterCategory;
    const int cat_count = var_layout->getCategoryCount();

    std::uint32_t set_index = 0;
    std::uint32_t binding_index = 0;
    bool has_descriptor = false;

    for (int ci = 0; ci < cat_count; ++ci) {
        auto const cat = var_layout->getCategoryByIndex(ci);
        switch (cat) {
            /* case C::DescriptorTableSpace:
                 set_index = static_cast<std::uint32_t>(var_layout->getOffset(cat));
                 has_descriptor = true;
                 break;
                 */
            case C::DescriptorTableSlot:
                binding_index = static_cast<std::uint32_t>(var_layout->getOffset(cat));
                has_descriptor = true;
                break;
            default:
                break;
        }
    }

    if (!has_descriptor) {
        return;
    }

    auto set_key = std::to_string(set_index);
    auto &set = entry.descriptor_sets[set_key];
    if (set.name.empty()) {
        set.name = "set" + std::to_string(set_index);
        set.index = set_index;
    }

    ReflectionData::BindingInfo bi{};
    bi.name = param_name.empty() ? ("binding" + std::to_string(binding_index)) : param_name;

    bi.binding = binding_index;
    bi.set_index = set_index;

    // Array size (1 if not an array)
    auto *type = type_layout->getType();
    std::uint32_t array_size = 1;
    if (type && type->getKind() == slang::TypeReflection::Kind::Array) {
        array_size = static_cast<std::uint32_t>(type->getElementCount());
    }
    bi.array_size = array_size;

    if (type && type->getName()) {
        bi.type_name = type->getName();
    }

    bi.resource_kind = resource_kind_from_type(type_layout);

    set.bindings.emplace(bi.name, std::move(bi));
}

// Reflect one *entry point* (stage) into ReflectionData
inline auto reflect_entry_point(ReflectionData &out, slang::EntryPointReflection *entry_point) -> void {
    if (!entry_point) {
        return;
    }

    auto const stage = entry_point->getStage();
    auto const stage_sv = stage_to_string(stage);
    std::string stage_key{stage_sv};

    auto &e = out.entries[stage_key];

    if (auto const *n = entry_point->getName()) {
        e.name = n;
    } else {
        e.name = stage_key;
    }
    e.stage_name = stage_key;
    e.stage_flags.insert(stage_key);

    const u32 param_count = entry_point->getParameterCount();
    for (auto i = 0U; i < param_count; ++i) {
        auto *param_layout = entry_point->getParameterByIndex(i);
        reflect_parameter_into_entry(e, param_layout);
    }
}

// Reflect a linked Slang program (IComponentType) into ReflectionData
inline auto reflect_program(Slang::ComPtr<slang::IComponentType> const &program, int target_index = 0)
        -> ReflectionData {
    ReflectionData result{};

    if (!program) {
        return result;
    }

    slang::ProgramLayout *layout = program->getLayout(target_index);
    if (!layout) {
        return result;
    }

    const u64 entry_count = layout->getEntryPointCount();
    for (auto i = 0U; i < entry_count; ++i) {
        auto *ep = layout->getEntryPointByIndex(i);
        reflect_entry_point(result, ep);
    }

    // You can optionally also reflect global-scope parameters here via
    // layout->getGlobalParamsVarLayout(), and fold them into each entry
    // if you want, but for now we keep it simple.

    return result;
}
