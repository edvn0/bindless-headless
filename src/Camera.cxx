#include "Camera.hxx"

#include <algorithm>

auto EditorCamera::view_matrix() const -> glm::mat4 { return glm::lookAtRH(position, position + forward, up); }

auto EditorCamera::camera_position() const -> glm::vec3 { return position; }

auto EditorCamera::set_from_orbit() -> void {
    // Convert yaw/pitch to forward
    const float yaw = glm::radians(yaw_deg);
    const float pitch = glm::radians(pitch_deg);

    glm::vec3 f{};
    f.x = glm::cos(pitch) * glm::cos(yaw);
    f.y = glm::sin(pitch);
    f.z = glm::cos(pitch) * glm::sin(yaw);
    forward = glm::normalize(f);

    right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
    up = glm::normalize(glm::cross(right, forward));

    // Orbit camera sits back along forward from pivot
    position = pivot - forward * distance;
}

auto EditorCamera::set_from_fly() -> void {
    const float yaw = glm::radians(yaw_deg);
    const float pitch = glm::radians(pitch_deg);

    glm::vec3 f{};
    f.x = glm::cos(pitch) * glm::cos(yaw);
    f.y = glm::sin(pitch);
    f.z = glm::cos(pitch) * glm::sin(yaw);
    forward = glm::normalize(f);

    right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
    up = glm::normalize(glm::cross(right, forward));
}

auto EditorCamera::clamp_pitch() -> void {
    // Avoid gimbal flip
    pitch_deg = std::clamp(pitch_deg, -89.5f, 89.5f);
}

auto EditorCamera::apply_orbit(const glm::vec2 delta_px) -> void {
    yaw_deg += delta_px.x * orbit_sensitivity;
    pitch_deg += -delta_px.y * orbit_sensitivity;
    clamp_pitch();
    set_from_orbit();
}

auto EditorCamera::apply_pan(const glm::vec2 delta_px) -> void {
    // Pan in camera plane; scale with distance to feel consistent.
    const float scale = (distance * 0.0015f) * pan_sensitivity;
    pivot += (-right * delta_px.x + up * delta_px.y) * scale;
    set_from_orbit();
}

auto EditorCamera::apply_dolly(float scroll_y) -> void {
    // Exponential zoom feels “editor-like”
    const float zoom = expf(-scroll_y * dolly_sensitivity);
    distance = std::clamp(distance * zoom, 0.05f, 50000.0f);
    set_from_orbit();
}

auto EditorCamera::apply_fly_look(const glm::vec2 delta_px) -> void {
    yaw_deg += delta_px.x * fly_look_sensitivity;
    pitch_deg += -delta_px.y * fly_look_sensitivity;
    clamp_pitch();
    set_from_fly();
}

auto EditorCamera::update_fly_move(GLFWwindow *window, double dt, bool fast) -> void {
    const float speed = (fast ? fly_speed_fast : fly_speed) * static_cast<float>(dt);

    constexpr auto key = [&](auto *w, int k) { return glfwGetKey(w, k) == GLFW_PRESS; };

    glm::vec3 move{0.0f};
    if (key(window, GLFW_KEY_W))
        move += forward;
    if (key(window, GLFW_KEY_S))
        move -= forward;
    if (key(window, GLFW_KEY_D))
        move += right;
    if (key(window, GLFW_KEY_A))
        move -= right;
    if (key(window, GLFW_KEY_E))
        move += glm::vec3(0.0f, 1.0f, 0.0f);
    if (key(window, GLFW_KEY_Q))
        move -= glm::vec3(0.0f, 1.0f, 0.0f);

    if (glm::length2(move) > 0.0f) {
        position += glm::normalize(move) * speed;
    }
}

static auto apply_deadzone(float v, float dz) -> float {
    if (std::abs(v) < dz)
        return 0.0f;
    return (v - std::copysign(dz, v)) / (1.0f - dz);
}

auto EditorCamera::update(GLFWwindow *window, double dt, CameraInput &in) -> void {
    // Determine modifiers via GLFW for robustness
    in.alt = (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS) ||
             (glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS);
    in.shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) ||
               (glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);
    in.ctrl = (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) ||
              (glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS);

    // Editor-ish mapping:
    //   Alt + LMB: orbit
    //   Alt + MMB: pan
    //   Scroll: dolly
    //   RMB (no alt): fly look + WASD/QE move
    if (orbit_mode) {
        if (in.scroll_delta != 0.0f) {
            apply_dolly(in.scroll_delta);
        }

        if (in.alt && in.lmb && (in.mouse_delta != glm::vec2(0.0f))) {
            apply_orbit(in.mouse_delta);
        } else if (in.alt && in.mmb && (in.mouse_delta != glm::vec2(0.0f))) {
            apply_pan(in.mouse_delta);
        }

        // Optional: RMB without Alt switches to fly temporarily
        if (!in.alt && in.rmb) {
            // Switch to fly with same yaw/pitch but keep position
            orbit_mode = false;
            set_from_fly();
        }
    }

    if (!orbit_mode) {
        if (in.rmb) {
            apply_fly_look(in.mouse_delta);
            update_fly_move(window, dt, in.shift);
        } else {
            // When RMB released, go back to orbit around current look point
            // Pick a pivot in front of camera so you “re-enter” orbit naturally.
            pivot = position + forward * std::max(distance, 1.0f);
            orbit_mode = true;
            set_from_orbit();
        }

        // While flying, scroll can adjust “distance” used when returning to orbit
        if (in.scroll_delta != 0.0f) {
            const float zoom = expf(-in.scroll_delta * dolly_sensitivity);
            distance = std::clamp(distance * zoom, 0.05f, 50000.0f);
        }
    }

    GLFWgamepadstate pad{};
    if (glfwGetGamepadState(GLFW_JOYSTICK_1, &pad)) {
        const float lx = apply_deadzone(pad.axes[GLFW_GAMEPAD_AXIS_LEFT_X], gamepad_deadzone);
        const float ly = apply_deadzone(pad.axes[GLFW_GAMEPAD_AXIS_LEFT_Y], gamepad_deadzone);
        const float rx = apply_deadzone(pad.axes[GLFW_GAMEPAD_AXIS_RIGHT_X], gamepad_deadzone);
        const float ry = apply_deadzone(pad.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y], gamepad_deadzone);
        // const float rt = (pad.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER] + 1.0f) * 0.5f; // [-1,1] -> [0,1]

        const float fdt = static_cast<float>(dt);
        const bool fast = pad.buttons[GLFW_GAMEPAD_BUTTON_LEFT_BUMPER] == GLFW_PRESS;
        const float speed = (fast ? fly_speed_fast : fly_speed) * fdt;

        // Right stick: look (fly mode) or orbit
        if (orbit_mode) {
            if (rx != 0.0f || ry != 0.0f) {
                apply_orbit({rx * gamepad_look_sensitivity * fdt, ry * gamepad_look_sensitivity * fdt});
            }
            // RT: dolly in/out via left stick Y
            if (ly != 0.0f) {
                apply_dolly(-ly * fdt * 5.0f);
            }
        } else {
            if (rx != 0.0f || ry != 0.0f) {
                apply_fly_look({rx * gamepad_look_sensitivity * fdt, ry * gamepad_look_sensitivity * fdt});
            }
            // Left stick: strafe/forward
            glm::vec3 move{0.0f};
            move += forward * (-ly);
            move += right * lx;
            // Bumpers: up/down
            if (pad.buttons[GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER] == GLFW_PRESS)
                move += glm::vec3(0.0f, 1.0f, 0.0f);
            if (pad.buttons[GLFW_GAMEPAD_BUTTON_A] == GLFW_PRESS)
                move -= glm::vec3(0.0f, 1.0f, 0.0f);

            if (glm::length2(move) > 0.0f)
                position += glm::normalize(move) * speed;
        }

        // B button: toggle orbit/fly
        if (pad.buttons[GLFW_GAMEPAD_BUTTON_B] == GLFW_PRESS) {
            orbit_mode = !orbit_mode;
            orbit_mode ? set_from_orbit() : set_from_fly();
        }
    }

    // Consume deltas
    in.mouse_delta = glm::vec2(0.0f);
    in.scroll_delta = 0.0f;
}
