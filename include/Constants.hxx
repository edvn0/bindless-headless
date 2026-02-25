#pragma once

#include "Types.hxx"

inline constexpr u32 white_texture_index = 0U;
inline constexpr u32 black_texture_index = 1U;
inline constexpr u32 normal_texture_index = 2U;

inline constexpr float fov_y = glm::radians(70.0f);
inline constexpr float z_near = 1.0f;
inline constexpr float z_far = 500.0f;

inline constexpr u32 max_lights_per_cluster = 128u;
inline constexpr u32 meters_per_unit_engine = 1u;
