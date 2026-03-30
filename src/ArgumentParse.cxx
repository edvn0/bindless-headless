#include "ArgumentParse.hxx"
#include "Types.hxx"

#include <lyra/help.hpp>
#include <lyra/val.hpp>
#include <volk.h>

#include <lyra/lyra.hpp>

auto build_base_cli(CLIOptions &opts) -> lyra::cli {
    return lyra::cli() | lyra::help(opts.show_help) | lyra::opt(opts.width, "w")["--width"]("Window width") |
           lyra::opt(opts.height, "h")["--height"]("Window height") | lyra::opt(opts.vsync)["--vsync"]("Enable vsync") |
           lyra::opt(opts.validation_layers)["--validation"]("Vulkan validation layers");
}
