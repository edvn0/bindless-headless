#include "Types.hxx"
#include <GLFW/glfw3.h>

auto OffscreenTarget::is_depth() const -> bool {
    return matches(format, VK_FORMAT_D16_UNORM, VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM_S8_UINT,
                   VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT);
}

auto OffscreenTarget::is_stencil() const -> bool {
    return matches(format, VK_FORMAT_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT,
                   VK_FORMAT_D32_SFLOAT_S8_UINT);
}


auto pipeline_cache_path() -> std::optional<std::filesystem::path> {
#if defined(_MSC_VER)
    // MSVC: use _dupenv_s
    char *buf{};
    size_t sz{};
    if (const auto ok = _dupenv_s(&buf, &sz, "BH_PIPE_CACHE_PATH") == 0 && buf; !ok)
        return std::nullopt;
    auto p = std::filesystem::path{buf};
    free(buf);
    return p;
#else
    // MinGW, GCC, Clang: use std::getenv
    const char *env_val = std::getenv("BH_PIPE_CACHE_PATH");
    if (!env_val)
        return std::nullopt;
    return std::filesystem::path{env_val};
#endif
}

auto OffscreenTarget::transition(VkCommandBuffer cmd, VkImageLayout old_layout, VkImageLayout new_layout,
                                 VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                                 VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
                                 VkImageSubresourceRange subresource_range) const -> void {
    VkImageMemoryBarrier2 barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext = nullptr,
            .srcStageMask = src_stage,
            .srcAccessMask = src_access,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_access,
            .oldLayout = old_layout,
            .newLayout = new_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = subresource_range,
    };

    VkDependencyInfo dep{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier,
    };

    vkCmdPipelineBarrier2(cmd, &dep);
}

auto OffscreenTarget::transition_if_not_initialised(VkCommandBuffer cmd, VkImageLayout new_layout,
                                                    std::pair<VkAccessFlags2, VkPipelineStageFlags2> destination_flags)
        -> void {
    if (initialized) [[likely]]
        return;

    VkImageAspectFlags aspect = 0;
    if (is_depth())
        aspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if (is_stencil())
        aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
    if (!is_depth() && !is_stencil())
        aspect = VK_IMAGE_ASPECT_COLOR_BIT;

    const VkImageSubresourceRange subresource_range{
            .aspectMask = aspect,
            .baseMipLevel = 0,
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };

    auto &&[dst_access, dst_stage] = destination_flags;

    const VkImageMemoryBarrier2 barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext = nullptr,
            .srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
            .srcAccessMask = VK_ACCESS_2_NONE,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_access,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = new_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = subresource_range,
    };

    const VkDependencyInfo dep{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier,
    };

    vkCmdPipelineBarrier2(cmd, &dep);

    initialized = true;
}


auto FrameStats::add_sample(double v) -> void {
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

auto FrameStats::quantile(double p) const -> double {
    if (count == 0)
        return 0.0;
    if (p <= 0.0)
        return min;
    if (p >= 1.0)
        return max;

    ensure_sorted();

    const double x = p * static_cast<double>(count - 1);
    const auto i = static_cast<std::size_t>(std::floor(x));
    const std::size_t j = std::min(i + 1, count - 1);
    const double t = x - static_cast<double>(i);

    return sorted[i] * (1.0 - t) + sorted[j] * t;
}

namespace {
    thread_local int s_profiler_depth = 0;
}

NanoProfiler::NanoProfiler(const std::string_view n) :
    start_time(glfwGetTime()), scope_name(n), depth(s_profiler_depth++) {}

NanoProfiler::~NanoProfiler() { end(); }

auto NanoProfiler::end() -> void {
    if (has_ended)
        return;
    --s_profiler_depth;
    auto end_time = glfwGetTime();
    auto millis = (end_time - start_time) * 1000.0;
    info("{:{}}"
         "Scope [{}] took {:4.2f} ms",
         "", depth * 2, scope_name.c_str(), millis);
    has_ended = true;
}
