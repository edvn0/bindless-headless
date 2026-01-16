#include "Logger.hxx"

#include <cstdlib>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#ifdef _WIN32
#include <stdlib.h>
#endif

namespace detail {

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

            std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink, error_sink};
            logger_ = std::make_shared<spdlog::logger>("app_logger", sinks.begin(), sinks.end());
            logger_->set_level(spdlog::level::trace);

            spdlog::register_logger(logger_);
        }

        ~LoggerImpl() { spdlog::drop_all(); }

        auto log(std::string_view msg, Level level) -> void {
            switch (level) {
                case Level::trace:
                    logger_->trace(msg);
                    break;
                case Level::debug:
                    logger_->debug(msg);
                    break;
                case Level::info:
                    logger_->info(msg);
                    break;
                case Level::warn:
                    logger_->warn(msg);
                    break;
                case Level::error:
                    logger_->error(msg);
                    break;
                case Level::critical:
                    logger_->critical(msg);
                    break;
            }
        }

    private:
        std::shared_ptr<spdlog::logger> logger_;

        static auto get_log_directory() -> std::string {
#ifdef _WIN32
            char *buffer = nullptr;
            size_t size = 0;
            if (_dupenv_s(&buffer, &size, "LOG_DIR") == 0 && buffer != nullptr) {
                std::string result(buffer);
                free(buffer);
                return result;
            }
#else
            if (const char *env_dir = std::getenv("LOG_DIR")) {
                return std::string(env_dir);
            }
#endif
            return "logs"; // default
        }
    };

    Logger::Logger() : impl_(std::make_unique<LoggerImpl>()) {}

    Logger::~Logger() = default;

    auto Logger::instance() -> Logger & {
        static Logger instance;
        return instance;
    }

    auto Logger::log(const std::string_view msg, const Level level) const -> void { impl_->log(msg, level); }

} // namespace detail
