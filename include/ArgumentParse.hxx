#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string_view>

struct CLIOptions {
    std::optional<std::filesystem::path> pipeline_cache_dir;
    std::uint32_t iteration_count = 5;
    std::uint32_t width{1280};
    std::uint32_t height{720};
    std::uint32_t light_count{50'000};
    bool vsync{false};
    std::uint32_t msaa{1};
    std::optional<bool> validation_layers{};
    bool disable_output_images{true};
    std::optional<std::string> title{"Bindless Headless"};
};

auto parse_cli(int argc, char **argv) -> std::optional<CLIOptions>;
