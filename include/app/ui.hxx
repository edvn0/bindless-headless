#pragma once

#include "app/app_forward.hxx"

#include "Types.hxx"

auto draw_ui(AppContext &, u32 frame_index, AppState &output) -> void;
struct UiFrameResult {
    VkExtent2D window_extent{};
    VkExtent2D desired_scene_extent{};
    bool minimized{false};
};
auto run_ui_frame(AppContext &ctx) -> UiFrameResult;
auto window_center(GLFWwindow *w) -> glm::vec2;
auto begin_cursor_capture(GLFWwindow *w, AppState &app) -> void;
auto end_cursor_capture(GLFWwindow *w, AppState &app) -> void;
