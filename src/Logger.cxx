#include "Logger.hxx"

#include <filesystem>

#include <cstdlib>
#include <mutex>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#ifdef _WIN32
#include <stdlib.h>
#endif

namespace detail {

    class ImGuiLogSink final : public spdlog::sinks::base_sink<std::mutex> {
    public:
        explicit ImGuiLogSink(std::shared_ptr<LogBuffer> buf) : m_buf{std::move(buf)} {}

    protected:
        auto sink_it_(const spdlog::details::log_msg &msg) -> void override {
            auto buf = m_buf.lock();
            if (!buf)
                return;

            spdlog::memory_buf_t formatted{};
            base_sink::formatter_->format(msg, formatted);

            auto str = fmt::to_string(formatted);
            if (!str.empty() && str.back() == '\n')
                str.pop_back();

            buf->push(LogEntry{to_level(msg.level), std::move(str)});
        }

        auto flush_() -> void override {}


    private:
        static auto to_level(spdlog::level::level_enum l) -> Level {
            using S = spdlog::level::level_enum;
            switch (l) {
                case S::trace:
                    return Level::trace;
                case S::debug:
                    return Level::debug;
                case S::info:
                    return Level::info;
                case S::warn:
                    return Level::warn;
                case S::err:
                    return Level::error;
                case S::critical:
                    return Level::critical;
                default:
                    return Level::info;
            }
        }

        std::weak_ptr<LogBuffer> m_buf;
    };

    class LoggerImpl {
    public:
        LoggerImpl() {
            auto log_dir = get_log_directory();

            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

            auto file_sink =
                    std::make_shared<spdlog::sinks::rotating_file_sink_mt>(log_dir + "/app.log", 1024 * 1024 * 5, 3);
            file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");

            auto error_sink =
                    std::make_shared<spdlog::sinks::rotating_file_sink_mt>(log_dir + "/error.log", 1024 * 1024 * 5, 3);
            error_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
            error_sink->set_level(spdlog::level::warn);

            gui_buffer = std::make_shared<LogBuffer>();
            auto imgui_sink = std::make_shared<ImGuiLogSink>(gui_buffer);
            imgui_sink->set_pattern("[%H:%M:%S.%e] [%l] %v");

            std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink, error_sink, imgui_sink};
            current_logger = std::make_shared<spdlog::logger>("app_logger", sinks.begin(), sinks.end());
            current_logger->set_level(spdlog::level::trace);

            spdlog::register_logger(current_logger);
        }

        ~LoggerImpl() { spdlog::drop_all(); }

        auto log(std::string_view msg, Level level) -> void {
            switch (level) {
                case Level::trace:
                    current_logger->trace(msg);
                    break;
                case Level::debug:
                    current_logger->debug(msg);
                    break;
                case Level::info:
                    current_logger->info(msg);
                    break;
                case Level::warn:
                    current_logger->warn(msg);
                    break;
                case Level::error:
                    current_logger->error(msg);
                    break;
                case Level::critical:
                    current_logger->critical(msg);
                    break;
            }
        }

        auto imgui_buffer() -> LogBuffer * { return gui_buffer.get(); }

    private:
        std::shared_ptr<spdlog::logger> current_logger;
        std::shared_ptr<LogBuffer> gui_buffer;

        static auto get_log_directory() -> std::string {
#if defined(_MSC_VER)
            // MSVC: use _dupenv_s
            char *buf{};
            size_t sz{};
            if (const auto ok = _dupenv_s(&buf, &sz, "LOG_DIR") == 0 && buf; !ok)
                return "logs";
            auto p = std::filesystem::path{buf};
            free(buf);
            return p.string();
#else
            // MinGW, GCC, Clang: use std::getenv
            const char *env_val = std::getenv("LOG_DIR");
            if (!env_val)
                return "logs";
            return env_val;
#endif
        }
    };

    Logger::Logger() : impl_(std::make_unique<LoggerImpl>()) {}

    Logger::~Logger() = default;

    auto Logger::imgui_buffer() -> LogBuffer * { return impl_->imgui_buffer(); }

    auto Logger::instance() -> Logger & {
        static Logger instance;
        return instance;
    }

    auto Logger::log(const std::string_view msg, const Level level) const -> void { impl_->log(msg, level); }

} // namespace detail

auto LogBuffer::push(LogEntry e) -> void {
    std::scoped_lock lock{mutex};
    entries.at(write_index) = std::move(e);
    is_dirty = true;
    write_index = (write_index + 1) % max_entries;
}
auto LogBuffer::clear() -> void {
    std::scoped_lock lock{mutex};
    entries.fill({});
    write_index = 0;
    is_dirty = false;
}
