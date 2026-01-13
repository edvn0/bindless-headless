#pragma once

#include <cstdint>
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
	bool initialized{ false };
};
