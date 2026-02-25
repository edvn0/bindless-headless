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
#include <vector>

#include <glm/glm.hpp>

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
    u32 flags{};

    constexpr auto set_albedo_map(bool has_map) -> void {
        if (has_map)
            flags |= FLAG_ALBEDO_MAP;
        else
            flags &= ~FLAG_ALBEDO_MAP;
    }
    constexpr auto set_normal_map(bool has_map) -> void {
        if (has_map)
            flags |= FLAG_NORMAL_MAP;
        else
            flags &= ~FLAG_NORMAL_MAP;
    }
    constexpr auto set_roughness_map(bool has_map) -> void {
        if (has_map)
            flags |= FLAG_ROUGHNESS_MAP;
        else
            flags &= ~FLAG_ROUGHNESS_MAP;
    }
    constexpr auto set_metallic_map(bool has_map) -> void {
        if (has_map)
            flags |= FLAG_METALLIC_MAP;
        else
            flags &= ~FLAG_METALLIC_MAP;
    }
    constexpr auto set_occlusion_map(bool has_map) -> void {
        if (has_map)
            flags |= FLAG_OCCLUSION_MAP;
        else
            flags &= ~FLAG_OCCLUSION_MAP;
    }
    constexpr auto set_emissive_map(bool has_map) -> void {
        if (has_map)
            flags |= FLAG_EMISSIVE_MAP;
        else
            flags &= ~FLAG_EMISSIVE_MAP;
    }
    constexpr auto set_is_alpha_tested(bool is_alpha) -> void {
        if (is_alpha)
            flags |= FLAG_ALPHA_TESTED;
        else
            flags &= ~FLAG_ALPHA_TESTED;
    }

    constexpr auto has_albedo_map() const -> bool { return (flags & FLAG_ALBEDO_MAP) != 0; }
    constexpr auto has_normal_map() const -> bool { return (flags & FLAG_NORMAL_MAP) != 0; }
    constexpr auto has_roughness_map() const -> bool { return (flags & FLAG_ROUGHNESS_MAP) != 0; }
    constexpr auto has_metallic_map() const -> bool { return (flags & FLAG_METALLIC_MAP) != 0; }
    constexpr auto has_occlusion_map() const -> bool { return (flags & FLAG_OCCLUSION_MAP) != 0; }
    constexpr auto has_emissive_map() const -> bool { return (flags & FLAG_EMISSIVE_MAP) != 0; }
    constexpr auto is_alpha_tested() const -> bool { return (flags & FLAG_ALPHA_TESTED) != 0; }

    constexpr static u32 FLAG_ALBEDO_MAP = 1 << 0;
    constexpr static u32 FLAG_NORMAL_MAP = 1 << 1;
    constexpr static u32 FLAG_ROUGHNESS_MAP = 1 << 2;
    constexpr static u32 FLAG_METALLIC_MAP = 1 << 3;
    constexpr static u32 FLAG_OCCLUSION_MAP = 1 << 4;
    constexpr static u32 FLAG_EMISSIVE_MAP = 1 << 5;
    constexpr static u32 FLAG_ALPHA_TESTED = 1 << 6;
};
