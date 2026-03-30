#pragma once

#include <ostream>
#include <streambuf>
#include <vector>
#include "Numeric.hxx"
#include "Types.hxx"

class TeeStreambuf : public std::streambuf {
public:
    explicit TeeStreambuf(usize count);
    auto emplace(std::streambuf *buf) -> void { sinks.emplace_back(buf); }

protected:
    auto overflow(int c) -> int override;
    auto xsputn(const char *, std::streamsize) -> std::streamsize override;
    auto sync() -> int override;

private:
    std::vector<std::streambuf *> sinks;
    NanoProfiler write_time;
};


auto write_scene_multi(const std::vector<std::byte> &data, const std::vector<std::filesystem::path> &destinations)
        -> bool;
