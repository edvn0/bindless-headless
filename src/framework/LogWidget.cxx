#include "framework/LogWidget.hxx"

#include "Assert.hxx"

LogWidget::LogWidget(LogBuffer *buf) : m_buf{buf} {

    if (!m_buf) {
        info("Defaulting to global log buffer");
        m_buf = detail::Logger::instance().imgui_buffer();
    }

    ASSERT(m_buf, "Buffer must be valid!");
}

auto LogWidget::level_color(Level l) -> ImVec4 {
    switch (l) {
        case Level::trace:
            return {0.6f, 0.6f, 0.6f, 1.f};
        case Level::debug:
            return {0.5f, 0.8f, 1.0f, 1.f};
        case Level::info:
            return {1.0f, 1.0f, 1.0f, 1.f};
        case Level::warn:
            return {1.0f, 0.8f, 0.2f, 1.f};
        case Level::error:
            return {1.0f, 0.3f, 0.3f, 1.f};
        case Level::critical:
            return {1.0f, 0.0f, 0.5f, 1.f};
    }
    return {1, 1, 1, 1};
}

auto LogWidget::draw(const char *title, bool *p_open) -> void {
    if (!ImGui::Begin(title, p_open)) {
        ImGui::End();
        return;
    }

    ImGui::Checkbox("Auto-scroll", &m_buf->auto_scroll);
    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
        std::scoped_lock lock{m_buf->mutex};
        m_buf->clear();
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.f);
    ImGui::Combo("##level", &m_min_level, k_level_names.data(), static_cast<int>(k_level_names.size()));
    ImGui::SameLine();
    m_filter.Draw("##filter", -1.f);
    ImGui::Separator();

    ImGui::BeginChild("##log_scroll", {0, 0}, ImGuiChildFlags_None);
    {
        std::scoped_lock lock{m_buf->mutex};
        for (const auto &e: m_buf->read_entries()) {
            if (static_cast<int>(e.level) < m_min_level)
                continue;
            if (!m_filter.PassFilter(e.message.c_str()))
                continue;

            ImGui::PushStyleColor(ImGuiCol_Text, level_color(e.level));
            ImGui::TextWrapped("%s", e.message.c_str());
            ImGui::PopStyleColor();
        }

        if (m_buf->auto_scroll && m_buf->is_dirty) {
            ImGui::SetScrollHereY(1.0f);
        }

        m_buf->is_dirty = false;
    }
    ImGui::EndChild();

    ImGui::End();
}
