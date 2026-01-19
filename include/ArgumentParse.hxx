#include <CLI/CLI.hpp>

#include <filesystem>
#include <optional>
#include <cstdint>
#include <cstdlib>

static auto env_pipeline_cache_dir() -> std::optional<std::filesystem::path> {
    char* buf{};
    size_t sz{};
    if (const auto ok = _dupenv_s(&buf, &sz, "BH_PIPE_CACHE_PATH") == 0 && buf; !ok) {
        return std::nullopt;
    }
    std::filesystem::path p{buf};
    free(buf);
    if (p.empty()) return std::nullopt;
    return p;
}

struct CLIOptions {
    std::optional<std::filesystem::path> pipeline_cache_dir;
    std::filesystem::path legacy_positional_dir{};
    std::uint32_t iteration_count = 5;
    std::uint32_t width {1280};
    std::uint32_t height {720};
    std::uint32_t light_count {50000};
    bool vsync {false};
};

static auto parse_cli(int argc, char** argv) -> CLIOptions {
    CLIOptions opt{};

    CLI::App app{"Bindless headless runner"};

    // New explicit flags
    std::filesystem::path flag_cache_dir{};
    app.add_option("--pipeline-cache-path", flag_cache_dir,
                   "Directory for bindless-headless.cache (overrides positional/env)")
        ->check(CLI::ExistingDirectory);

    app.add_option("-n,--iterations", opt.iteration_count,
                   "Number of main frame iterations")
        ->check(CLI::Range(1u, 10000000u));

    // Back-compat positional: first non-flag argument
    app.add_option("pipeline_cache_dir", opt.legacy_positional_dir,
                   "Legacy positional cache directory (used if no --pipeline-cache-path)")
        ->check(CLI::ExistingDirectory);

    app.add_option("-l,--light_count", opt.light_count, "Light count");
    app.add_option("--width", opt.width, "Width of 'window'")->default_val(1280);
    app.add_option("--height", opt.height, "Height of 'window'")->default_val(720);
    app.add_option("-s,--vsync", opt.vsync, "Vsync'")->default_val(true);

    // Let CLI11 handle -h/--help
    app.allow_extras(false);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        // exits with help/error message already emitted by CLI11
        std::exit(app.exit(e));
    }

    // Precedence: flag > positional > env
    if (!flag_cache_dir.empty()) {
        opt.pipeline_cache_dir = flag_cache_dir;
    } else if (!opt.legacy_positional_dir.empty()) {
        opt.pipeline_cache_dir = opt.legacy_positional_dir;
    } else {
        opt.pipeline_cache_dir = env_pipeline_cache_dir();
    }

    return opt;
}
