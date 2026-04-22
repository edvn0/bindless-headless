#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <lyra/cli.hpp>
#include <optional>
#include <string_view>

struct CLIOptions {
    std::uint32_t width{1280};
    std::uint32_t height{720};
    bool vsync{true};
    bool validation_layers{true};
    std::optional<std::string> title{};
    std::optional<std::filesystem::path> pipeline_cache_dir{};
    bool show_help{false};
};

struct EngineOptions : CLIOptions {
    std::uint32_t iteration_count{5};
    std::uint32_t light_count{50'000};
    std::uint32_t msaa{1};
    bool disable_output_images{true};
};

auto build_base_cli(CLIOptions &opts) -> lyra::cli;
