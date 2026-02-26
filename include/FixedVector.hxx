#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "Assert.hxx"

template<typename T, std::size_t Capacity>
class FixedVector {
public:
    using value_type             = T;
    using size_type              = std::size_t;
    using reference              = T&;
    using const_reference        = const T&;
    using pointer                = T*;
    using const_pointer          = const T*;
    using iterator               = T*;
    using const_iterator         = const T*;
    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    constexpr FixedVector() noexcept = default;

    constexpr FixedVector(std::initializer_list<T> init) {
        for (auto&& v : init)
            push_back(v);
    }

    constexpr explicit FixedVector(size_type count, const T& value = T{}) {
        ASSERT(count <= Capacity, "FixedVector: initializer count exceeds capacity");
        for (size_type i = 0; i < count; ++i)
            push_back(value);
    }

    constexpr FixedVector(const FixedVector& other) {
        for (const auto& v : other)
            emplace_back(v);
    }

    constexpr FixedVector(FixedVector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
        for (auto& v : other)
            emplace_back(std::move(v));
        other.clear();
    }

    constexpr auto operator=(const FixedVector& other) -> FixedVector& {
        if (this != &other) {
            clear();
            for (const auto& v : other)
                emplace_back(v);
        }
        return *this;
    }

    constexpr auto operator=(FixedVector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) -> FixedVector& {
        if (this != &other) {
            clear();
            for (auto& v : other)
                emplace_back(std::move(v));
            other.clear();
        }
        return *this;
    }

    constexpr ~FixedVector() { clear(); }

    // --- Element access ---

    [[nodiscard]] constexpr auto at(size_type i) noexcept -> reference {
        if (i >= m_size) std::abort();
        return data()[i];
    }

    [[nodiscard]] constexpr auto at(size_type i) const -> const_reference {
        if (i >= m_size) std::abort();
        return data()[i];
    }

    [[nodiscard]] constexpr auto operator[](size_type i) noexcept -> reference            { ASSERT(i < m_size, "FixedVector::operator[]: index out of bounds"); return data()[i]; }
    [[nodiscard]] constexpr auto operator[](size_type i) const noexcept -> const_reference { ASSERT(i < m_size, "FixedVector::operator[]: index out of bounds"); return data()[i]; }

    [[nodiscard]] constexpr auto front() noexcept -> reference            { ASSERT(!empty(), "FixedVector::front: empty vector"); return data()[0]; }
    [[nodiscard]] constexpr auto front() const noexcept -> const_reference { ASSERT(!empty(), "FixedVector::front: empty vector"); return data()[0]; }

    [[nodiscard]] constexpr auto back() noexcept -> reference             { ASSERT(!empty(), "FixedVector::back: empty vector"); return data()[m_size - 1]; }
    [[nodiscard]] constexpr auto back() const noexcept -> const_reference  { ASSERT(!empty(), "FixedVector::back: empty vector"); return data()[m_size - 1]; }

    [[nodiscard]] constexpr auto data() noexcept -> pointer             { return std::launder(reinterpret_cast<T*>(m_storage)); }
    [[nodiscard]] constexpr auto data() const noexcept -> const_pointer { return std::launder(reinterpret_cast<const T*>(m_storage)); }

    [[nodiscard]] constexpr auto as_span() noexcept -> std::span<T>             { return {data(), m_size}; }
    [[nodiscard]] constexpr auto as_span() const noexcept -> std::span<const T> { return {data(), m_size}; }

    // --- Iterators ---

    [[nodiscard]] constexpr auto begin() noexcept -> iterator { return data(); }
    [[nodiscard]] constexpr auto end()   noexcept -> iterator { return data() + m_size; }

    [[nodiscard]] constexpr auto begin()  const noexcept -> const_iterator { return data(); }
    [[nodiscard]] constexpr auto end()    const noexcept -> const_iterator { return data() + m_size; }
    [[nodiscard]] constexpr auto cbegin() const noexcept -> const_iterator { return begin(); }
    [[nodiscard]] constexpr auto cend()   const noexcept -> const_iterator { return end(); }

    [[nodiscard]] constexpr auto rbegin() noexcept -> reverse_iterator { return reverse_iterator(end()); }
    [[nodiscard]] constexpr auto rend()   noexcept -> reverse_iterator { return reverse_iterator(begin()); }

    [[nodiscard]] constexpr auto rbegin()  const noexcept -> const_reverse_iterator { return const_reverse_iterator(end()); }
    [[nodiscard]] constexpr auto rend()    const noexcept -> const_reverse_iterator { return const_reverse_iterator(begin()); }
    [[nodiscard]] constexpr auto crbegin() const noexcept -> const_reverse_iterator { return rbegin(); }
    [[nodiscard]] constexpr auto crend()   const noexcept -> const_reverse_iterator { return rend(); }

    // --- Capacity ---

    [[nodiscard]] constexpr auto empty()    const noexcept -> bool      { return m_size == 0; }
    [[nodiscard]] constexpr auto size()     const noexcept -> size_type { return m_size; }
    [[nodiscard]] constexpr auto capacity() const noexcept -> size_type { return Capacity; }
    [[nodiscard]] constexpr auto full()     const noexcept -> bool      { return m_size == Capacity; }

    template<typename... Args>
    constexpr auto emplace_back(Args&&... args) -> reference {
        ASSERT(m_size < Capacity, "FixedVector: capacity exceeded");
        T* slot = data() + m_size;
        std::construct_at(slot, std::forward<Args>(args)...);
        ++m_size;
        return *slot;
    }

    constexpr auto push_back(const T& value) -> void { emplace_back(value); }
    constexpr auto push_back(T&& value)      -> void { emplace_back(std::move(value)); }

    constexpr auto pop_back() noexcept -> void {
        ASSERT(!empty(), "FixedVector::pop_back: empty vector");
        std::destroy_at(data() + --m_size);
    }

    constexpr auto erase(const_iterator pos) -> iterator {
        ASSERT(pos >= cbegin() && pos < cend(), "FixedVector::erase: position out of range");
        auto* target = const_cast<iterator>(pos);
        std::move(target + 1, end(), target);
        pop_back();
        return target;
    }

    constexpr auto erase(const_iterator first, const_iterator last) -> iterator {
        ASSERT(first >= cbegin() && last <= cend() && first <= last, "FixedVector::erase: range is out of bounds");
        auto* f       = const_cast<iterator>(first);
        auto* l       = const_cast<iterator>(last);
        auto* new_end = std::move(l, end(), f);
        std::destroy(new_end, end());
        m_size -= static_cast<size_type>(last - first);
        return f;
    }

    constexpr auto clear() noexcept -> void {
        std::destroy(begin(), end());
        m_size = 0;
    }

    constexpr auto resize(size_type new_size, const T& value = T{}) -> void {
        ASSERT(new_size <= Capacity, "FixedVector::resize: new size exceeds capacity");
        while (m_size > new_size) pop_back();
        while (m_size < new_size) push_back(value);
    }


    [[nodiscard]] constexpr auto operator==(const FixedVector& other) const noexcept -> bool {
        return std::equal(begin(), end(), other.begin(), other.end());
    }

    [[nodiscard]] constexpr auto operator<=>(const FixedVector& other) const noexcept {
        return std::lexicographical_compare_three_way(begin(), end(), other.begin(), other.end());
    }

private:
    alignas(T) std::byte m_storage[Capacity * sizeof(T)]{};
    size_type m_size{0};
};