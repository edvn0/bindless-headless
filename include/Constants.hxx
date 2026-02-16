#pragma once

#include "Types.hxx"

inline constexpr u32 white_texture_index = 0U;
inline constexpr u32 black_texture_index = 1U;
inline constexpr u32 normal_texture_index = 2U;

inline constexpr float fov_y = glm::radians(70.0f);
inline constexpr float z_near = 1.0f;
inline constexpr float z_far =10000.0f;
