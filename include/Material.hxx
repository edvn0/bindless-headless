#pragma once

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <future>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <tl/expected.hpp>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include "Bitfield.hxx"
#include "Numeric.hxx"

struct MaterialData {
    std::string name{};
    std::string albedo_map{};
    glm::vec4 albedo_factor{1.0f};
    std::string normal_map{};
    std::string roughness_map{};
    float roughness_factor{1.0f};
    std::string metallic_map{};
    float metallic_factor{1.0f};
    std::string occlusion_map{};
    std::string emissive_map{};
    glm::vec3 emissive_factor{0.0f};
    bool is_alpha_tested{false};
};

enum class MaterialFlags : u32 {
    None = 0,
    Albedo = 1 << 0,
    Normal = 1 << 1,
    Roughness = 1 << 2,
    Metallic = 1 << 3,
    Occlusion = 1 << 4,
    Emissive = 1 << 5,
    AlphaTested = 1 << 6,
    All = 0xFFFFFFFF
};
MAKE_BITFIELD(MaterialFlags)

struct GPUMaterialData {
    u32 albedo_map{};
    glm::vec4 albedo_factor{1.0f};
    u32 normal_map{};
    u32 roughness_map{};
    float roughness_factor{1.0f};
    u32 metallic_map{};
    float metallic_factor{1.0f};
    u32 occlusion_map{};
    u32 emissive_map{};
    glm::vec3 emissive_factor{0.0f};
    MaterialFlags flags;

    constexpr auto set_albedo_map(bool has_map) -> void {
        if (has_map)
            flags |= MaterialFlags::Albedo;
        else
            flags &= ~MaterialFlags::Albedo;
    }
    constexpr auto set_normal_map(bool has_map) -> void {
        if (has_map)
            flags |= MaterialFlags::Normal;
        else
            flags &= ~MaterialFlags::Normal;
    }
    constexpr auto set_roughness_map(bool has_map) -> void {
        if (has_map)
            flags |= MaterialFlags::Roughness;
        else
            flags &= ~MaterialFlags::Roughness;
    }
    constexpr auto set_metallic_map(bool has_map) -> void {
        if (has_map)
            flags |= MaterialFlags::Metallic;
        else
            flags &= ~MaterialFlags::Metallic;
    }
    constexpr auto set_occlusion_map(bool has_map) -> void {
        if (has_map)
            flags |= MaterialFlags::Occlusion;
        else
            flags &= ~MaterialFlags::Occlusion;
    }
    constexpr auto set_emissive_map(bool has_map) -> void {
        if (has_map)
            flags |= MaterialFlags::Emissive;
        else
            flags &= ~MaterialFlags::Emissive;
    }
    constexpr auto set_is_alpha_tested(bool is_alpha) -> void {
        if (is_alpha)
            flags |= MaterialFlags::AlphaTested;
        else
            flags &= ~MaterialFlags::AlphaTested;
    }

    constexpr auto has_albedo_map() const -> bool { return has(flags, MaterialFlags::Albedo); }
    constexpr auto has_normal_map() const -> bool { return has(flags, MaterialFlags::Normal); }
    constexpr auto has_roughness_map() const -> bool { return has(flags, MaterialFlags::Roughness); }
    constexpr auto has_metallic_map() const -> bool { return has(flags, MaterialFlags::Metallic); }
    constexpr auto has_occlusion_map() const -> bool { return has(flags, MaterialFlags::Occlusion); }
    constexpr auto has_emissive_map() const -> bool { return has(flags, MaterialFlags::Emissive); }
    constexpr auto is_alpha_tested() const -> bool { return has(flags, MaterialFlags::AlphaTested); }
};
