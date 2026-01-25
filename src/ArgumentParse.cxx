#include "ArgumentParse.hxx"
#include "Types.hxx"

#include <volk.h>

 auto env_pipeline_cache_dir() -> std::optional<std::filesystem::path> {
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



 auto parse_cli(int argc, char** argv) -> CLIOptions {
    CLIOptions opt{};

    CLI::App app{"Bindless headless runner"};

    std::filesystem::path flag_cache_dir{};
    app.add_option("--pipeline-cache-path", flag_cache_dir,
                   "Directory for bindless-headless.cache (overrides positional/env)")
        ->check(CLI::ExistingDirectory);

    app.add_option("-n,--iterations", opt.iteration_count,
                   "Number of main frame iterations")
        ->check(CLI::Range(1u, 10000000u));

    app.add_option("pipeline_cache_dir", opt.legacy_positional_dir,
                   "Legacy positional cache directory (used if no --pipeline-cache-path)")
        ->check(CLI::ExistingDirectory);

    app.add_option("-l,--light_count", opt.light_count, "Light count");
    app.add_option("--width", opt.width, "Width of 'window'")->default_val(1280);
    app.add_option("--height", opt.height, "Height of 'window'")->default_val(720);
    app.add_option("--vsync", opt.vsync, "Vsync'")->default_val(true);
std::string new_cwd{};
     app.add_option("--cwd", new_cwd, "Set the current working directory")->default_val(std::string {});
app.add_option("--msaa", opt.msaa, "MSAA sample count (1,2,4,8,16,32,64)")
        ->default_val(1)
        ->check(CLI::IsMember({1u, 2u, 4u, 8u, 16u, 32u, 64u}));
    app.allow_extras(false);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
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

     std::filesystem::path current {std::filesystem::current_path()};
     if (!new_cwd.empty()) {
         current = std::filesystem::absolute(new_cwd);
     }
     std::filesystem::current_path(current);

    return opt;
}
