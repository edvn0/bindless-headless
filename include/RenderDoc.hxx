#pragma once

#include <cstdint>
#include <string_view>

struct RenderDocApi;

struct RenderDocContext {
    RenderDocApi *api{nullptr}; // null ↔ RenderDoc not present / not loaded

    [[nodiscard]] auto is_active() const -> bool { return api != nullptr; }

    [[nodiscard]] auto is_capturing() const -> bool;

    auto begin_frame_capture(void *vk_instance, void *wnd_handle = nullptr) const -> void;
    auto end_frame_capture(void *vk_instance, void *wnd_handle = nullptr) const -> void;

    auto trigger_capture() const -> void;

    [[nodiscard]] auto launch_replay_ui() const -> bool;

    auto set_capture_path(std::string_view path_template) const -> void;
};

[[nodiscard]] auto renderdoc_init() -> RenderDocContext;
