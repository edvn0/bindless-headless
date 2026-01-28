#include <CLI/CLI.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string_view>

#include "Logger.hxx"

struct CLIOptions {
    std::optional<std::filesystem::path> pipeline_cache_dir;
    std::uint32_t iteration_count = 5;
    std::uint32_t width{1280};
    std::uint32_t height{720};
    std::uint32_t light_count{50'000};
    bool vsync{false};
    std::uint32_t msaa{1};
};

auto parse_cli(int argc, char **argv) -> CLIOptions;
