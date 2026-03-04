#pragma once

#include "Types.hxx"

#include "scene/Forward.hxx"

struct MeshSubmission {
    u32 mesh_index; // Inside of the renderer - the index of the DrawID really.
    glm::mat4x3 transform;
    u32 material_id;
};

struct RenderQueue {
    std::vector<MeshSubmission> meshes;
    usize high_watermark = 0;
    usize frames_below_watermark = 0;

    static constexpr usize shrink_threshold_frames = 120;
    static constexpr float shrink_factor = 0.75f;

    auto submit(MeshSubmission&& sub) -> void { meshes.emplace_back(sub); }

    auto flush() -> std::span<const MeshSubmission> { return meshes; }

    auto reset() -> void {
        const usize current = meshes.size();

        if (current > high_watermark) {
            high_watermark = current;
            frames_below_watermark = 0;
        } else if (current < static_cast<usize>(static_cast<float>(high_watermark) * shrink_factor)) {
            ++frames_below_watermark;

            if (frames_below_watermark >= shrink_threshold_frames) {
                const usize new_cap = static_cast<usize>(static_cast<float>(current) * 1.25f) + 8;
                meshes.shrink_to_fit();
                meshes.reserve(new_cap);
                high_watermark = new_cap;
                frames_below_watermark = 0;
            }
        } else {
            frames_below_watermark = 0;
        }

        meshes.clear();
    }
};

// Per frame work - submits meshes to the renderer.
auto submit_mesh_instances(Scene &scene, RenderQueue &queue) -> void;