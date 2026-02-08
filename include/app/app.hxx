#pragma once

#include <tl/expected.hpp>
#include "ArgumentParse.hxx"
#include "Error.hxx"

struct InstanceWithDebug;

class BindlessApp {
public:
    auto run(CLIOptions &opts, InstanceWithDebug &instance) -> tl::expected<int, Error>;
};
