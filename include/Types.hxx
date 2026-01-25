#pragma once

#include <cstdint>
#include <mutex>
#include <numeric>
#include <string_view>
#include <volk.h>

#include <vk_mem_alloc.h>

using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using u8 = std::uint8_t;
using i8 = std::int8_t;

inline constexpr u32 frames_in_flight = 3; // renderer-side DAG cycle
inline constexpr u32 max_in_flight = 2; // GPU submit throttle depth

struct string_hash {
    using is_transparent = void;

    auto operator()(std::string_view v) const noexcept -> std::size_t { return std::hash<std::string_view>{}(v); }
    auto operator()(std::string const &s) const noexcept -> std::size_t { return (*this)(std::string_view{s}); }
    auto operator()(char const *s) const noexcept -> std::size_t { return (*this)(std::string_view{s}); }
};

struct string_eq {
    using is_transparent = void;
    auto operator()(std::string_view a, std::string_view b) const noexcept -> bool { return a == b; }
};


enum class DeviceAddress : std::uint64_t {
    Invalid = 0,
};

auto vk_check(VkResult result) -> void;

struct OffscreenTarget {
    VkImage image{};
    VkImageView sampled_view{};
    VkImageView storage_view{};
    VkImageView attachment_view{};
    VkFormat format{};
    VmaAllocation allocation{};
    u32 width{};
    u32 height{};
    bool initialized{false};

    auto is_depth() const -> bool;
    auto is_stencil() const -> bool;
    auto transition_if_not_initialised(VkCommandBuffer, VkImageLayout,
                                       std::pair<VkAccessFlagBits, VkPipelineStageFlagBits> destination_flags) -> void;
};

struct FrameStats {
    std::vector<double> samples;

    std::size_t count = 0;
    double mean = 0.0;
    double m2 = 0.0;
    double sum = 0.0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();

    mutable std::vector<double> sorted;
    mutable bool sorted_dirty = true;

    explicit FrameStats(std::size_t capacity = 0) {
        if (capacity) {
            samples.reserve(capacity);
            sorted.reserve(capacity);
        }
    }

    auto clear() -> void {
        samples.clear();
        sorted.clear();
        sorted_dirty = true;

        count = 0;
        mean = 0.0;
        m2 = 0.0;
        sum = 0.0;
        min = std::numeric_limits<double>::infinity();
        max = -std::numeric_limits<double>::infinity();
    }

    auto reserve(std::size_t capacity) -> void {
        samples.reserve(capacity);
        sorted.reserve(capacity);
    }

    auto add_sample(double v) -> void;

    auto total() const -> double { return (count == 0) ? 0.0 : sum; }
    auto avg() const -> double { return (count == 0) ? 0.0 : mean; }

    auto variance_pop() const -> double {
        if (count < 2)
            return 0.0;
        return m2 / static_cast<double>(count);
    }

    auto stddev_pop() const -> double { return std::sqrt(variance_pop()); }

    auto variance_sample() const -> double {
        if (count < 2)
            return 0.0;
        return m2 / static_cast<double>(count - 1);
    }

    auto stddev_sample() const -> double { return std::sqrt(variance_sample()); }

private:
    auto ensure_sorted() const -> void {
        if (!sorted_dirty)
            return;
        sorted = samples;
        std::ranges::sort(sorted);
        sorted_dirty = false;
    }

public:
    // Linear-interpolated quantile (p in [0, 1])
    auto quantile(double p) const -> double;

    auto median() const -> double { return quantile(0.5); }
    auto p90() const -> double { return quantile(0.90); }
    auto p95() const -> double { return quantile(0.95); }
    auto p99() const -> double { return quantile(0.99); }

    struct Quartiles {
        double q1 = 0.0;
        double q2 = 0.0;
        double q3 = 0.0;
        double iqr = 0.0;
    };

    auto quartiles() const -> Quartiles {
        Quartiles q;
        if (count == 0)
            return q;
        q.q1 = quantile(0.25);
        q.q2 = quantile(0.50);
        q.q3 = quantile(0.75);
        q.iqr = q.q3 - q.q1;
        return q;
    }
};

template<typename T>
concept IsFunctionPointerLike =
        std::is_pointer_v<std::remove_cvref_t<T>> && std::is_function_v<std::remove_pointer_t<std::remove_cvref_t<T>>>;
/*
template<IsFunctionPointerLike Fn>
class MaybeNoOp {
    mutable std::mutex access_mutex;
    Fn f = nullptr;

public:
    explicit MaybeNoOp(Fn fn) : f(std::move(fn)) {}
    explicit MaybeNoOp(std::nullptr_t) : f({}) {}
    MaybeNoOp() = default;

    [[nodiscard]] auto empty() const noexcept -> bool {
        std::lock_guard lock(access_mutex);
        return f == nullptr;
    }

    explicit operator bool() const noexcept {
        return !empty();
    }

    auto operator=(Fn fn) noexcept -> MaybeNoOp& {
        std::lock_guard lock(access_mutex);
        f = fn;
        return *this;
    }

    auto operator=(std::nullptr_t) noexcept -> MaybeNoOp& {
        std::lock_guard lock(access_mutex);
        f = nullptr;
        return *this;
    }

    template<typename... Args>
    auto operator()(Args &&... args) const {
        using r_t = std::invoke_result_t<Fn, Args...>;

        std::unique_lock lock(access_mutex);  // Lock BEFORE checking f

        if constexpr (std::is_void_v<r_t>) {
            if (f) {
                std::invoke(f, std::forward<Args>(args)...);
                return true;
            }
            return false;
        } else {
            if (f) {
                return std::optional<r_t>{std::invoke(f, std::forward<Args>(args)...)};
            }
            return std::optional<r_t>{};
        }
    }
};
*/

template<IsFunctionPointerLike Fn>
class MaybeNoOp {
    std::atomic<Fn> f;

public:
    explicit MaybeNoOp(Fn fn) : f(fn) {}
    explicit MaybeNoOp(std::nullptr_t) : f(nullptr) {}
    MaybeNoOp() : f(nullptr) {}

    [[nodiscard]] auto empty() const noexcept -> bool { return f.load(std::memory_order_acquire) == nullptr; }

    explicit operator bool() const noexcept { return !empty(); }

    auto operator=(Fn fn) noexcept -> MaybeNoOp & {
        f.store(fn, std::memory_order_release);
        return *this;
    }

    auto operator=(std::nullptr_t) noexcept -> MaybeNoOp & {
        f.store(nullptr, std::memory_order_release);
        return *this;
    }

    template<typename... Args>
    auto operator()(Args &&...args) const {
        using r_t = std::invoke_result_t<Fn, Args...>;

        // Load the function pointer atomically
        Fn fn_copy = f.load(std::memory_order_acquire);

        if constexpr (std::is_void_v<r_t>) {
            if (fn_copy) {
                std::invoke(fn_copy, std::forward<Args>(args)...);
                return true;
            }
            return false;
        } else {
            if (fn_copy) {
                return std::optional<r_t>{std::invoke(fn_copy, std::forward<Args>(args)...)};
            }
            return std::optional<r_t>{};
        }
    }
};

constexpr auto matches(const auto &needle, const auto &&...haystack) { return ((needle == haystack) || ...); }

namespace std {
    template<>
    struct formatter<FrameStats::Quartiles> : formatter<double> {
        auto format(const FrameStats::Quartiles &q, auto &ctx) const {
            using std::format_to;
            format_to(ctx.out(), "Q1: {:.3f}, Q2: {:.3f}, Q3: {:.3f}, IQR: {:.3f}", q.q1, q.q2, q.q3, q.iqr);
            return ctx.out();
        }
    };
} // namespace std
