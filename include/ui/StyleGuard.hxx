#include <imgui.h>
#include <tuple>

#include "Numeric.hxx"

struct StyleGuard {
    i32 count = 0;

    template<typename... Args>
        requires(sizeof...(Args) > 0)
    StyleGuard(Args &&...args) : count(sizeof...(args)) {
        (apply_push(std::forward<Args>(args)), ...);
    }

    ~StyleGuard() {
        if (count > 0) {
            ImGui::PopStyleVar(count);
        }
    }

    StyleGuard(const StyleGuard &) = delete;
    StyleGuard &operator=(const StyleGuard &) = delete;
    StyleGuard(StyleGuard &&) = delete;
    StyleGuard &operator=(StyleGuard &&) = delete;


private:
    template<typename T>
    auto apply_push(const std::pair<ImGuiStyleVar_, T> &pair) {
        ImGui::PushStyleVar(pair.first, pair.second);
    }
};
