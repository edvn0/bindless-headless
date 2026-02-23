#pragma once

#include <GLFW/glfw3.h>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include "Numeric.hxx"

struct CameraInput {
    bool lmb{false};
    bool mmb{false};
    bool rmb{false};
    bool alt{false};
    bool shift{false};
    bool ctrl{false};

    bool orbit_capture{false}; // NEW

    glm::vec2 gamepad_left{0.0f};
    glm::vec2 gamepad_right{0.0f};
    f32 gamepad_rt{0.0f};
    bool gamepad_lb{false};
    bool gamepad_rb{false};

    // for edge detection
    bool gamepad_b_prev{false};

    glm::vec2 mouse_delta{0.0f, 0.0f}; // pixels
    f32 scroll_delta{0.0f}; // y scroll units
};

struct EditorCamera {
    f32 gamepad_look_sensitivity = 120.0f; // degrees per second
    f32 gamepad_move_sensitivity = 5.0f;
    f32 gamepad_deadzone = 0.15f;
    // Orbit target
    glm::vec3 pivot{0.0f, 0.0f, 0.0f};

    // Orbit state (degrees)
    f32 yaw_deg{-90.0f};
    f32 pitch_deg{-20.0f};
    f32 distance{35.0f};

    // Fly state
    glm::vec3 position{15.0f, 10.0f, -20.0f};

    // Derived basis
    glm::vec3 forward{0.0f, 0.0f, -1.0f};
    glm::vec3 right{1.0f, 0.0f, 0.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};

    // Tunables
    f32 orbit_sensitivity{0.12f}; // deg / pixel
    f32 pan_sensitivity{1.0f}; // unitless, scaled by distance
    f32 dolly_sensitivity{0.12f}; // exponential zoom
    f32 fly_look_sensitivity{0.12f}; // deg / pixel

    f32 fly_speed{10.0f};
    f32 fly_speed_fast{35.0f};

    // If true, position is driven by orbit around pivot. If false, classic fly cam.
    bool orbit_mode{true};

    auto view_matrix() const -> glm::mat4;

    auto camera_position() const -> glm::vec3;

    auto set_from_orbit() -> void;
    auto set_from_fly() -> void;

    auto clamp_pitch() -> void;

    auto apply_orbit(const glm::vec2 delta_px) -> void;
    auto apply_pan(const glm::vec2 delta_px) -> void;
    auto apply_dolly(f32 scroll_y) -> void;
    auto apply_fly_look(const glm::vec2 delta_px) -> void;

    auto update_fly_move(GLFWwindow *window, double dt, bool fast) -> void;
    auto update(GLFWwindow *window, double dt, CameraInput &in) -> void;
};
