#pragma once

#include <cstdlib>
#include <source_location>

#include "Logger.hxx"

// ---------------------------------------------------------------------------
// ASSERT(cond)           -- always-evaluated condition, fires in debug builds
// ASSERT(cond, msg)      -- with a string literal message
// ASSERT_UNREACHABLE()   -- marks logically unreachable code paths
// ---------------------------------------------------------------------------

namespace detail {

    [[noreturn]] inline void assert_fail(const char *condition, const char *message,
                                         std::source_location loc = std::source_location::current()) noexcept {
        if (message) {
            critical("Assertion failed: {}\n  Message:  {}\n  Function: {}\n  File:     {}:{}", condition, message,
                     loc.function_name(), loc.file_name(), loc.line());
        } else {
            critical("Assertion failed: {}\n  Function: {}\n  File:     {}:{}", condition, loc.function_name(),
                     loc.file_name(), loc.line());
        }
        std::abort();
    }

    [[noreturn]] inline void unreachable_fail(std::source_location loc = std::source_location::current()) noexcept {
        critical("Unreachable code reached\n  Function: {}\n  File:     {}:{}", loc.function_name(), loc.file_name(),
                 loc.line());
        std::abort();
    }

} // namespace detail

#ifndef NDEBUG

#define ASSERT(cond, ...)                                                                                              \
    do {                                                                                                               \
        if (!(cond)) [[unlikely]] {                                                                                    \
            ::detail::assert_fail(#cond, _ASSERT_MSG_OR_NULL(__VA_ARGS__), std::source_location::current());           \
        }                                                                                                              \
    } while (false)

#define ASSERT_UNREACHABLE() ::detail::unreachable_fail(std::source_location::current())

#else

#define ASSERT(cond, ...)                                                                                              \
    do {                                                                                                               \
        (void) (cond);                                                                                                 \
    } while (false)
#define ASSERT_UNREACHABLE() __builtin_unreachable()

#endif

#define _ASSERT_MSG_OR_NULL(...) _ASSERT_MSG_OR_NULL_IMPL(__VA_ARGS__, nullptr)
#define _ASSERT_MSG_OR_NULL_IMPL(msg, ...) (msg)
