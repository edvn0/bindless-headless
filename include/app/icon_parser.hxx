#pragma once

#include <tl/expected.hpp>
#include <volk.h>

#include "Types.hxx"

// Grammar:
//   icon_filename ::= name '.' format '.' oetf
//   name          ::= [a-zA-Z0-9_-]+
//   format        ::= 'r' | 'rg' | 'rgb' | 'rgba'
//   oetf          ::= 'linear' | 'srgb'

struct IconLoadDescription {
    enum class Channels { r, rg, rgb, rgba };

    Channels channels = Channels::rgba;
    bool srgb = false;

    [[nodiscard]] auto vk_format() const -> VkFormat {
        switch (channels) {
            case Channels::r:
                return VK_FORMAT_R8_UNORM;
            case Channels::rg:
                return VK_FORMAT_R8G8_UNORM;
            case Channels::rgb:
                return srgb ? VK_FORMAT_R8G8B8_SRGB : VK_FORMAT_R8G8B8_UNORM;
            case Channels::rgba:
                return srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
        }
        return VK_FORMAT_R8G8B8A8_UNORM;
    }

    [[nodiscard]] auto bytes_per_pixel() const -> int {
        switch (channels) {
            case Channels::r:
                return 1;
            case Channels::rg:
                return 2;
            case Channels::rgb:
                return 3;
            case Channels::rgba:
                return 4;
        }
        return 4;
    }
};


struct IconFilenameParser {
    explicit IconFilenameParser(std::string_view input) : input_{input}, pos_{0} {}

    struct ParseError {
        std::string message;
        usize pos;
    };

    struct Result {
        std::string name;
        IconLoadDescription desc;
    };

    auto parse() -> tl::expected<Result, ParseError> {
        auto name = parse_name();
        if (!name)
            return tl::unexpected{name.error()};

        if (!consume('.'))
            return tl::unexpected{ParseError{"expected '.' after name", pos_}};

        auto format = parse_format();
        if (!format)
            return tl::unexpected{format.error()};

        if (!consume('.'))
            return tl::unexpected{ParseError{"expected '.' after format", pos_}};

        auto oetf = parse_oetf();
        if (!oetf)
            return tl::unexpected{oetf.error()};

        if (!at_end())
            return tl::unexpected{ParseError{"unexpected trailing characters", pos_}};

        return Result{
                .name = std::move(*name),
                .desc = IconLoadDescription{.channels = *format, .srgb = *oetf},
        };
    }

private:
    std::string_view input_;
    usize pos_;

    [[nodiscard]] auto peek() const -> std::optional<char> {
        if (pos_ >= input_.size())
            return std::nullopt;
        return input_[pos_];
    }

    [[nodiscard]] auto at_end() const -> bool { return pos_ >= input_.size(); }

    auto consume(char expected) -> bool {
        if (peek() == expected) {
            ++pos_;
            return true;
        }
        return false;
    }

    auto consume_while(auto pred) -> std::string_view {
        const usize start = pos_;
        while (!at_end() && pred(input_[pos_]))
            ++pos_;
        return input_.substr(start, pos_ - start);
    }

    auto expect_literal(std::string_view lit) -> bool {
        if (input_.substr(pos_, lit.size()) == lit) {
            pos_ += lit.size();
            return true;
        }
        return false;
    }

    auto parse_name() -> tl::expected<std::string, ParseError> {
        const auto sv = consume_while([](char c) { return std::isalnum(static_cast<u8>(c)) || c == '-' || c == '_'; });
        if (sv.empty())
            return tl::unexpected{ParseError{"expected icon name", pos_}};
        return std::string{sv};
    }

    auto parse_format() -> tl::expected<IconLoadDescription::Channels, ParseError> {
        if (expect_literal("rgba"))
            return IconLoadDescription::Channels::rgba;
        if (expect_literal("rgb"))
            return IconLoadDescription::Channels::rgb;
        if (expect_literal("rg"))
            return IconLoadDescription::Channels::rg;
        if (expect_literal("r"))
            return IconLoadDescription::Channels::r;
        return tl::unexpected{ParseError{"expected format: 'r', 'rg', 'rgb', or 'rgba'", pos_}};
    }

    auto parse_oetf() -> tl::expected<bool, ParseError> {
        if (expect_literal("linear"))
            return false;
        if (expect_literal("srgb"))
            return true;
        return tl::unexpected{ParseError{"expected oetf: 'linear' or 'srgb'", pos_}};
    }
};
