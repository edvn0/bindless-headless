#pragma once

#include <cstdint>
#include <numeric>
#include <vulkan/vulkan.h>

#include <vma/vk_mem_alloc.h>

using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using u8 = std::uint8_t;
using i8 = std::int8_t;

auto vk_check(VkResult result) -> void;

struct OffscreenTarget {
    VkImage image{};
    VkImageView sampled_view{};
    VkImageView storage_view{};
    VkFormat format{};
    VmaAllocation allocation{};
    u32 width{};
    u32 height{};
    bool initialized{false};
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

    auto add_sample(double v) -> void {
        samples.push_back(v);
        sorted_dirty = true;

        ++count;
        sum += v;
        min = std::min(min, v);
        max = std::max(max, v);

        // Welford
        const double delta = v - mean;
        mean += delta / static_cast<double>(count);
        const double delta2 = v - mean;
        m2 += delta * delta2;
    }

    auto total() const -> double { return (count == 0) ? 0.0 : sum; }
    auto avg() const -> double { return (count == 0) ? 0.0 : mean; }

    auto variance_pop() const -> double {
        if (count < 2) return 0.0;
        return m2 / static_cast<double>(count);
    }

    auto stddev_pop() const -> double {
        return std::sqrt(variance_pop());
    }

    auto variance_sample() const -> double {
        if (count < 2) return 0.0;
        return m2 / static_cast<double>(count - 1);
    }

    auto stddev_sample() const -> double {
        return std::sqrt(variance_sample());
    }

private:
    auto ensure_sorted() const -> void {
        if (!sorted_dirty) return;
        sorted = samples;
        std::ranges::sort(sorted);
        sorted_dirty = false;
    }

public:
    // Linear-interpolated quantile (p in [0, 1])
    auto quantile(double p) const -> double {
        if (count == 0) return 0.0;
        if (p <= 0.0) return min;
        if (p >= 1.0) return max;

        ensure_sorted();

        const double x = p * static_cast<double>(count - 1);
        const std::size_t i = static_cast<std::size_t>(std::floor(x));
        const std::size_t j = std::min(i + 1, count - 1);
        const double t = x - static_cast<double>(i);

        return sorted[i] * (1.0 - t) + sorted[j] * t;
    }

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
        if (count == 0) return q;
        q.q1 = quantile(0.25);
        q.q2 = quantile(0.50);
        q.q3 = quantile(0.75);
        q.iqr = q.q3 - q.q1;
        return q;
    }
};

namespace std {
    template<> struct std::formatter<FrameStats::Quartiles> : std::formatter<double> {
        auto format(const FrameStats::Quartiles& q, auto& ctx) const {
            using std::format_to;
            format_to(ctx.out(), "Q1: {:.3f}, Q2: {:.3f}, Q3: {:.3f}, IQR: {:.3f}",
                      q.q1, q.q2, q.q3, q.iqr);
            return ctx.out();
        }
    };
}