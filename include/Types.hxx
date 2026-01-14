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

    auto add_sample(double v) -> void {
        samples.push_back(v);
    }

    auto mean() const -> double {
        if (samples.empty()) return 0.0;
        const auto sum = std::accumulate(samples.begin(), samples.end(), 0.0);
        return sum / static_cast<double>(samples.size());
    }

    auto median() const -> double {
        if (samples.empty()) return 0.0;
        auto tmp = samples;
        std::ranges::sort(tmp);
        const auto n = tmp.size();
        if ((n & 1u) == 0u)
            return (tmp[n / 2 - 1] + tmp[n / 2]) * 0.5;
        return tmp[n / 2];
    }

    auto stddev() const -> double {
        if (samples.size() < 2) return 0.0;
        const auto m = mean();
        double acc = 0.0;
        for (const auto& v: samples) acc += (v - m) * (v - m);
        return std::sqrt(acc / static_cast<double>(samples.size()));
    }
};