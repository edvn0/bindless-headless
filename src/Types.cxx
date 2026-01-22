#include "Types.hxx"

auto OffscreenTarget::is_depth() const -> bool {
    return matches(format, VK_FORMAT_D16_UNORM, VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM_S8_UINT,
                   VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT);
}

auto OffscreenTarget::is_stencil() const -> bool {
    return matches(format, VK_FORMAT_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT,
                   VK_FORMAT_D32_SFLOAT_S8_UINT);
}

auto OffscreenTarget::transition_if_not_initialised(
        VkCommandBuffer cmd, VkImageLayout new_layout,
        std::pair<VkAccessFlagBits, VkPipelineStageFlagBits> destination_flags) -> void {
    if (initialized) [[likely]]
        return;

    // Fixed aspect mask logic
    VkImageAspectFlags aspect = 0;
    if (is_depth())
        aspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if (is_stencil())
        aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
    if (!is_depth() && !is_stencil())
        aspect = VK_IMAGE_ASPECT_COLOR_BIT;

    VkImageSubresourceRange subresource_range = {
            .aspectMask = aspect,
            .baseMipLevel = 0,
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };

    auto &&[dst_access, dst_stage] = destination_flags;

    VkImageMemoryBarrier depth_barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = 0,
            .dstAccessMask = static_cast<VkAccessFlags>(dst_access),
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = new_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = subresource_range,
    };

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, static_cast<VkPipelineStageFlags>(dst_stage), 0, 0,
                         nullptr, 0, nullptr, 1, &depth_barrier);

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
