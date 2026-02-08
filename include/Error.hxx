#pragma once

#include "Numeric.hxx"

#include <source_location>

struct Error {
    enum class Type : u32 {
        MeshLoadError,
        TextureLoadError,
        ShaderCompileError,
        ShaderLinkError,
        RenderError,
        InvalidArgument,
        DeviceSelectionError,
        CouldNotMapMemory,
        CouldNotCreateBuffer,
        FileNotFoundError,
        InvalidSize,
        UnknownError
    };

    Type type;
    std::string message;
    std::source_location location{std::source_location::current()};

    template<typename... Ts>
        requires(sizeof...(Ts) > 0)
    static auto make_error(Type type, std::format_string<Ts...> fmt, Ts &&...args) -> Error {
        return make_error(type, std::source_location::current(), fmt, std::forward<Ts>(args)...);
    }

    template<typename... Ts>
        requires(sizeof...(Ts) > 0)
    static auto make_error(Type type, std::source_location location, std::format_string<Ts...> fmt, Ts &&...args)
            -> Error {
        return make_error_impl(type, std::format(fmt, std::forward<Ts>(args)...), location);
    }

    static auto make_error(Type type, const std::string &data,
                           std::source_location location = std::source_location::current()) -> Error {
        return make_error_impl(type, data, location);
    }

private:
    static auto make_error_impl(Type type, const std::string &message, std::source_location location) -> Error {
        return Error{.type = type, .message = message, .location = location};
    }
};
