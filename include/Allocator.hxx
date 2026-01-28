#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>

#if defined(HAS_TRACY)
#include <tracy/Tracy.hpp> // provides TracyAlloc/TracyFree (+ Secure variants)
#endif

template<class T, bool Secure = false>
struct tracy_allocator {
    using value_type = T;

    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    tracy_allocator() noexcept = default;

    template<class U>
    tracy_allocator(const tracy_allocator<U, Secure> &) noexcept {}

    [[nodiscard]] auto allocate(std::size_t n) -> T * {
        // Delegate actual allocation to std::allocator (which typically uses ::operator new).
        T *p = std::allocator<T>{}.allocate(n);

#if defined(HAS_TRACY)
        if (p != nullptr && n != 0) {
            const std::size_t bytes = n * sizeof(T);
            if constexpr (Secure) {
                TracySecureAlloc(p, bytes);
            } else {
                TracyAlloc(p, bytes);
            }
        }
#endif

        return p;
    }

    auto deallocate(T *p, std::size_t n) noexcept -> void {
#if defined(HAS_TRACY)
        // Tracy requires a corresponding alloc/free pair for each tracked address.
        // Don't emit events for null pointers.
        if (p != nullptr) {
            if constexpr (Secure) {
                TracySecureFree(p);
            } else {
                TracyFree(p);
            }
        }
#endif

        // std::allocator::deallocate requires the same n that was used in allocate.
        std::allocator<T>{}.deallocate(p, n);
    }

    template<class U>
    struct rebind {
        using other = tracy_allocator<U, Secure>;
    };

    // Stateless allocators compare equal.
    friend auto operator==(const tracy_allocator &, const tracy_allocator &) noexcept -> bool = default;
};

// Convenience alias: when Tracy is off, you get the normal allocator.
#if defined(HAS_TRACY)
template<class T>
using default_allocator = tracy_allocator<T, /*Secure=*/false>;
#else
template<class T>
using default_allocator = std::allocator<T>;
#endif
