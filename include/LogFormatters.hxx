#pragma once

#include <format>
#include <source_location>
#include "Error.hxx"
#include "Numeric.hxx"
#include "Types.hxx"

namespace std {

    template<>
    struct formatter<source_location> {
        constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

        auto format(const source_location &loc, format_context &ctx) const {
            return format_to(ctx.out(), "{}:{}:{}", loc.file_name(), loc.line(), loc.function_name());
        }
    };


    template<>
    struct formatter<VkFormat> {
        constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

        auto format(const VkFormat &loc, format_context &ctx) const {
            return format_to(ctx.out(), "{}", static_cast<u32>(loc));
        }
    };

    template<>
    struct formatter<filesystem::path> {
        constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

        auto format(const filesystem::path &t, format_context &ctx) const {
            return format_to(ctx.out(), "{}", t.string());
        }
    };

    template<std::integral T>
    struct formatter<std::atomic<T>> {
        constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }
        auto format(const std::atomic<T> &t, format_context &ctx) const { return format_to(ctx.out(), "{}", t.load()); }
    };

    template<>
    struct formatter<FrameStats::Quartiles> : formatter<double> {
        auto format(const FrameStats::Quartiles &q, auto &ctx) const {
            using std::format_to;
            format_to(ctx.out(), "Q1: {:.3f}, Q2: {:.3f}, Q3: {:.3f}, IQR: {:.3f}", q.q1, q.q2, q.q3, q.iqr);
            return ctx.out();
        }
    };

    template<>
    struct formatter<Error> : formatter<string_view> {
        auto format(const Error &err, format_context &ctx) const {
            std::string s = std::format("[{}] {} (at {}:{}:{})", error_to_string(err.type), err.message,
                                        err.location.file_name(), err.location.line(), err.location.column());
            return std::formatter<std::string_view>::format(s, ctx);
        }
    };
} // namespace std
