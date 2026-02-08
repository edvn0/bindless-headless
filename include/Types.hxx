#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <format>
#include <functional>
#include <mutex>
#include <numeric>
#include <source_location>
#include <string_view>
#include <volk.h>

#include "Error.hxx"
#include "Numeric.hxx"

#include <vk_mem_alloc.h>


inline constexpr u32 frames_in_flight = 3; // renderer-side DAG cycle
inline constexpr u32 max_in_flight = 2; // GPU submit throttle depth

struct Deleter {
    template<typename T>
    auto operator()(T *t) noexcept -> void {
        delete t;
    }
};

struct string_hash {
    using is_transparent = void;

    auto operator()(const std::string_view v) const noexcept -> std::size_t { return std::hash<std::string_view>{}(v); }
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

template<typename T>
struct TypedDeviceAddress {
    DeviceAddress address;

    explicit(false) operator DeviceAddress() const { return address; }
};

auto vk_check(VkResult result) -> void;

auto pipeline_cache_path() -> std::optional<std::filesystem::path>;

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
                                       std::pair<VkAccessFlagBits2, VkPipelineStageFlagBits2> destination_flags)
            -> void;

    auto transition(VkCommandBuffer cmd, VkImageLayout old_layout, VkImageLayout new_layout,
                    VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage,
                    VkAccessFlags2 dst_access, VkImageSubresourceRange subresource_range) const -> void;

    auto transition(VkCommandBuffer cmd, VkImageLayout old_layout, VkImageLayout new_layout,
                    VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage,
                    VkAccessFlags2 dst_access) const -> void {
        transition(cmd, old_layout, new_layout, src_stage, src_access, dst_stage, dst_access,
                   default_subresource_range());
    }

private:
    auto default_subresource_range() const -> VkImageSubresourceRange {
        VkImageAspectFlags aspect = 0;

        if (is_depth())
            aspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
        if (is_stencil())
            aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
        if (aspect == 0)
            aspect = VK_IMAGE_ASPECT_COLOR_BIT;

        return VkImageSubresourceRange{
                .aspectMask = aspect,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
        };
    }
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
        std::sort(sorted.begin(), sorted.end());
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


template<typename... Ts>
auto first_non_empty(const Ts &...strs) -> std::string {
    for (const auto *s: {&strs...}) {
        if (!s->empty())
            return *s;
    }
    return {};
}

#define TRY_UNWRAP_TO(var_name, expected_expr, msg)                                                                    \
    auto var_name##_tmp = (expected_expr);                                                                             \
    if (!var_name##_tmp.has_value()) {                                                                                 \
        const auto &err = var_name##_tmp.error();                                                                      \
        warn("{}: (Error Type: {}) {}", msg, static_cast<i32>(err.type), err.message);                                 \
        return err;                                                                                                    \
    }                                                                                                                  \
    auto var_name = std::move(var_name##_tmp.value());

#define TRY_UNWRAP_WITH_DISCARD(var_name, expected_expr, msg)                                                          \
    auto var_name##_tmp = (expected_expr);                                                                             \
    if (!var_name##_tmp.has_value()) {                                                                                 \
        const auto &err = var_name##_tmp.error();                                                                      \
        warn("{}: (Error Type: {}) {}", msg, static_cast<i32>(err.type), err.message);                                 \
        return;                                                                                                        \
    }                                                                                                                  \
    auto var_name = std::move(var_name##_tmp.value());

#define TRY_PROPAGATE(var_name, expected_expr, msg)                                                                    \
    auto var_name##_tmp = (expected_expr);                                                                             \
    if (!var_name##_tmp.has_value()) {                                                                                 \
        auto err = std::move(var_name##_tmp.error());                                                                  \
        warn("{}: {}", msg, err.message);                                                                              \
        return tl::make_unexpected(std::move(err));                                                                    \
    }                                                                                                                  \
    auto var_name = std::move(var_name##_tmp.value());

template<typename T>
concept IsFunctionPointerLike =
        std::is_pointer_v<std::remove_cvref_t<T>> && std::is_function_v<std::remove_pointer_t<std::remove_cvref_t<T>>>;

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

constexpr std::string_view to_string(Error::Type type) {
    using enum Error::Type;
    switch (type) {
        case MeshLoadError:
            return "Mesh Load Error";
        case TextureLoadError:
            return "Texture Load Error";
        case ShaderCompileError:
            return "Shader Compile Error";
        case ShaderLinkError:
            return "Shader Link Error";
        case RenderError:
            return "Render Error";
        case DeviceSelectionError:
            return "Device Selection Error";
        case UnknownError:
            return "Unknown Error";
        default:
            return "Invalid Error Type";
    }
}

namespace std {
    template<>
    struct formatter<FrameStats::Quartiles> : formatter<double> {
        auto format(const FrameStats::Quartiles &q, auto &ctx) const {
            using std::format_to;
            format_to(ctx.out(), "Q1: {:.3f}, Q2: {:.3f}, Q3: {:.3f}, IQR: {:.3f}", q.q1, q.q2, q.q3, q.iqr);
            return ctx.out();
        }
    };

    // Error
    template<>
    struct formatter<Error> : formatter<string_view> {
        auto format(const Error &err, format_context &ctx) const {
            std::string s = std::format("[{}] {} (at {}:{}:{})", to_string(err.type), err.message,
                                        err.location.file_name(), err.location.line(), err.location.column());
            return std::formatter<std::string_view>::format(s, ctx);
        }
    };
} // namespace std
