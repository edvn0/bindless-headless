#pragma once

#include <format>
#include <memory>
#include <string_view>
#include "Numeric.hxx"

enum class Level { trace, debug, info, warn, error, critical };

struct LogEntry {
    Level level;
    std::string message; // already formatted by spdlog pattern
};

struct LogBuffer {
    auto push(LogEntry) -> void;

    u32 write_index{0};
    std::mutex mutex;
    bool is_dirty{false};
    bool auto_scroll{true};

    auto read_entries() const { return std::span<const LogEntry>(entries.begin(), entries.begin() + write_index); }
    auto clear() -> void;

private:
    static constexpr u32 max_entries = 4096;
    std::array<LogEntry, max_entries> entries{};
};

namespace detail {

    class LoggerImpl;

    class Logger {
    public:
        static auto instance() -> Logger &;

        auto log(std::string_view msg, Level level) const -> void;

        template<typename... Args>
        auto log_formatted(Level level, std::format_string<Args...> fmt, Args &&...args) -> void {
            auto msg = std::format(fmt, std::forward<Args>(args)...);
            log(msg, level);
        }

        Logger(const Logger &) = delete;
        auto operator=(const Logger &) -> Logger & = delete;

        auto imgui_buffer() -> LogBuffer *;

    private:
        Logger();
        ~Logger();

        std::unique_ptr<LoggerImpl> impl_;
    };

} // namespace detail

// Public API - free functions

template<typename... Args>
auto trace(std::format_string<Args...> fmt, Args &&...args) -> void {
    detail::Logger::instance().log_formatted(Level::trace, fmt, std::forward<Args>(args)...);
}

template<typename... Args>
auto debug(std::format_string<Args...> fmt, Args &&...args) -> void {
    detail::Logger::instance().log_formatted(Level::debug, fmt, std::forward<Args>(args)...);
}

template<typename... Args>
auto info(std::format_string<Args...> fmt, Args &&...args) -> void {
    detail::Logger::instance().log_formatted(Level::info, fmt, std::forward<Args>(args)...);
}

template<typename... Args>
auto warn(std::format_string<Args...> fmt, Args &&...args) -> void {
    detail::Logger::instance().log_formatted(Level::warn, fmt, std::forward<Args>(args)...);
}

template<typename... Args>
auto error(std::format_string<Args...> fmt, Args &&...args) -> void {
    detail::Logger::instance().log_formatted(Level::error, fmt, std::forward<Args>(args)...);
}

template<typename... Args>
auto critical(std::format_string<Args...> fmt, Args &&...args) -> void {
    detail::Logger::instance().log_formatted(Level::critical, fmt, std::forward<Args>(args)...);
}

// Helper for custom log levels
inline auto log(std::string_view msg, Level level) -> void { detail::Logger::instance().log(msg, level); }
