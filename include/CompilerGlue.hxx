#include <concepts>
#include <iterator>
#include <ranges>
#include <vector>

namespace detail {
    template<typename Container, std::ranges::range Range>
    constexpr auto to_impl(Range &&range) -> Container {
        if constexpr (std::ranges::sized_range<Range>) {
            Container result;
            if constexpr (requires { result.reserve(std::ranges::size(range)); }) {
                result.reserve(std::ranges::size(range));
            }
            std::ranges::copy(range, std::back_inserter(result));
            return result;
        } else {
            Container result;
            std::ranges::copy(range, std::back_inserter(result));
            return result;
        }
    }

    template<typename Container>
    struct to_adaptor_closure {
        template<std::ranges::range Range>
        constexpr auto operator()(Range &&range) const -> Container {
            return to_impl<Container>(std::forward<Range>(range));
        }

        template<std::ranges::range Range>
        friend constexpr auto operator|(Range &&range, const to_adaptor_closure &closure) -> Container {
            return closure(std::forward<Range>(range));
        }
    };
} // namespace detail

template<typename Container>
constexpr auto to() -> detail::to_adaptor_closure<Container> {
    return {};
}

template<typename Container, std::ranges::range Range>
constexpr auto to(Range &&range) -> Container {
    return detail::to_impl<Container>(std::forward<Range>(range));
}
