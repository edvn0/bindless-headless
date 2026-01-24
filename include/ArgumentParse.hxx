#include <CLI/CLI.hpp>

#include <filesystem>
#include <optional>
#include <cstdint>
#include <cstdlib>
#include <string_view>

#include "Logger.hxx"

auto env_pipeline_cache_dir() -> std::optional<std::filesystem::path>;
struct CLIOptions {
    std::optional<std::filesystem::path> pipeline_cache_dir;
    std::filesystem::path legacy_positional_dir{};
    std::uint32_t iteration_count = 5;
    std::uint32_t width {1280};
    std::uint32_t height {720};
    std::uint32_t light_count {50000};
    bool vsync {false};
};

 auto parse_cli(int argc, char** argv) -> CLIOptions;
