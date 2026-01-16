#pragma once

#include <format>
#include <memory>
#include <string_view>

namespace detail {

    enum class Level { trace, debug, info, warn, error, critical };

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

    private:
        Logger();
        ~Logger();

        std::unique_ptr<LoggerImpl> impl_;
    };

} // namespace detail

// Public API - free functions

template<typename... Args>
auto trace(std::format_string<Args...> fmt, Args &&...args) -> void {
    detail::Logger::instance().log_formatted(detail::Level::trace, fmt, std::forward<Args>(args)...);
}

template<typename... Args>
auto debug(std::format_string<Args...> fmt, Args &&...args) -> void {
    detail::Logger::instance().log_formatted(detail::Level::debug, fmt, std::forward<Args>(args)...);
}

template<typename... Args>
auto info(std::format_string<Args...> fmt, Args &&...args) -> void {
    detail::Logger::instance().log_formatted(detail::Level::info, fmt, std::forward<Args>(args)...);
}

template<typename... Args>
auto warn(std::format_string<Args...> fmt, Args &&...args) -> void {
    detail::Logger::instance().log_formatted(detail::Level::warn, fmt, std::forward<Args>(args)...);
}

template<typename... Args>
auto error(std::format_string<Args...> fmt, Args &&...args) -> void {
    detail::Logger::instance().log_formatted(detail::Level::error, fmt, std::forward<Args>(args)...);
}

template<typename... Args>
auto critical(std::format_string<Args...> fmt, Args &&...args) -> void {
    detail::Logger::instance().log_formatted(detail::Level::critical, fmt, std::forward<Args>(args)...);
}

// Helper for custom log levels
inline auto log(std::string_view msg, detail::Level level) -> void { detail::Logger::instance().log(msg, level); }
