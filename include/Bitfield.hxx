#pragma once

#include <type_traits>
#include <utility>

#define MAKE_BITFIELD(E)                                                                                               \
    [[nodiscard]] constexpr E operator|(E l, E r) noexcept {                                                           \
        return static_cast<E>(std::to_underlying(l) | std::to_underlying(r));                                          \
    }                                                                                                                  \
    [[nodiscard]] constexpr E operator&(E l, E r) noexcept {                                                           \
        return static_cast<E>(std::to_underlying(l) & std::to_underlying(r));                                          \
    }                                                                                                                  \
    [[nodiscard]] constexpr E operator^(E l, E r) noexcept {                                                           \
        return static_cast<E>(std::to_underlying(l) ^ std::to_underlying(r));                                          \
    }                                                                                                                  \
    [[nodiscard]] constexpr E operator~(E e) noexcept { return static_cast<E>(~std::to_underlying(e)); }               \
    constexpr E &operator|=(E &l, E r) noexcept { return l = l | r; }                                                  \
    constexpr E &operator&=(E &l, E r) noexcept { return l = l & r; }                                                  \
    constexpr E &operator^=(E &l, E r) noexcept { return l = l ^ r; }                                                  \
    [[nodiscard]] constexpr bool any(E e) noexcept { return std::to_underlying(e) != 0; }                              \
    [[nodiscard]] constexpr bool has(E e, E other) noexcept { return any(e & other); }
