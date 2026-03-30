#pragma once

#include <format>
#include <source_location>
#include "Numeric.hxx"

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

} // namespace std
