#pragma once

#include "ArgumentParse.hxx"
#include "Error.hxx"
#include <tl/expected.hpp>

struct InstanceWithDebug;

class BindlessApp {
public:
    auto run(CLIOptions& opts, InstanceWithDebug& instance) -> tl::expected<int, Error>;
};
