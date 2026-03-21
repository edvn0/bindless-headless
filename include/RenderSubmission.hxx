#pragma once

#include "Types.hxx"

#include "scene/Forward.hxx"

#include <glm/mat4x3.hpp>
#include <span>
#include <type_traits>
#include <vector>

struct MeshSubmission {
    glm::mat4x3 transform;
    u32 mesh_index; // Inside of the renderer - the index of the DrawID really.
    u32 lod_level; // 0 = full detail
};

template<typename T>
struct WatermarkedQueue {
    static_assert(std::is_trivial_v<T>);

    std::vector<T> objects;
    usize high_watermark = 0;
    usize frames_below_watermark = 0;

    static constexpr usize shrink_threshold_frames = 120;
    static constexpr float shrink_factor = 0.75f;

    auto submit(T &&sub) -> void { objects.emplace_back(sub); }

    auto flush() -> std::span<const T> { return objects; }

    auto reset() -> void {
        const usize current = objects.size();

        if (current > high_watermark) {
            high_watermark = current;
            frames_below_watermark = 0;
        } else if (current < static_cast<usize>(static_cast<float>(high_watermark) * shrink_factor)) {
            ++frames_below_watermark;

            if (frames_below_watermark >= shrink_threshold_frames) {
                const usize new_cap = static_cast<usize>(static_cast<float>(current) * 1.25f) + 8;
                objects.shrink_to_fit();
                objects.reserve(new_cap);
                high_watermark = new_cap;
                frames_below_watermark = 0;
            }
        } else {
            frames_below_watermark = 0;
        }

        objects.clear();
    }
};

using RenderQueue = WatermarkedQueue<MeshSubmission>;

// Per frame work - submits meshes to the renderer.
auto submit_mesh_instances(Scene &scene, RenderQueue &queue) -> void;
