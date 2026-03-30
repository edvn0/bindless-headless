#pragma once

#include "Logger.hxx"

#include <imgui.h>

class LogWidget {
public:
    explicit LogWidget(LogBuffer * = nullptr);
    auto draw(const char *title, bool *p_open = nullptr) -> void;

private:
    static auto level_color(Level l) -> ImVec4;

    static constexpr std::array k_level_names{"trace", "debug", "info", "warn", "error", "critical"};

    LogBuffer *m_buf{nullptr};
    ImGuiTextFilter m_filter{};
    int m_min_level{static_cast<int>(Level::trace)};
};
