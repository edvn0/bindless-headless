#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "Types.hxx"

struct ReflectionData {
    struct PushConstantInfo {
        std::string name;
        u32 offset{};
        u32 size{};
        std::unordered_map<std::string, u32, string_hash, string_eq> fields;
    };

    struct BindingInfo {
        std::string name;
        u32 binding{};
        u32 array_size{1};
        std::string type_name;
        std::string resource_kind;
        u32 set_index{};
    };

    struct SetInfo {
        std::string name;
        u32 index{};
        std::unordered_map<std::string, BindingInfo, string_hash, string_eq> bindings;
    };

    struct EntryInfo {
        std::string name; // entry-point function name (backend-specific)
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


enum class ShaderStage {
    Vertex,
    Fragment,
    Compute,
    Geometry,
    Hull,
    Domain,
    RayGen,
    Intersection,
    AnyHit,
    ClosestHit,
    Miss,
    Callable,
    Unknown,
};

inline auto stage_to_string(ShaderStage s) -> std::string_view {
    switch (s) {
        case ShaderStage::Vertex:
            return "vertex";
        case ShaderStage::Fragment:
            return "fragment";
        case ShaderStage::Compute:
            return "compute";
        case ShaderStage::Geometry:
            return "geometry";
        case ShaderStage::Hull:
            return "hull";
        case ShaderStage::Domain:
            return "domain";
        case ShaderStage::RayGen:
            return "raygen";
        case ShaderStage::Intersection:
            return "intersection";
        case ShaderStage::AnyHit:
            return "any_hit";
        case ShaderStage::ClosestHit:
            return "closest_hit";
        case ShaderStage::Miss:
            return "miss";
        case ShaderStage::Callable:
            return "callable";
        default:
            return "unknown";
    }
}
