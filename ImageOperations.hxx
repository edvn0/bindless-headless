#pragma once

#include "Pool.hxx"

#include <vma/vk_mem_alloc.h>

#include <vulkan/vulkan.h>

#include <fstream>
#include <ostream>
#include <string_view>

namespace image_operations {
	auto write_to_disk(
		DestructionContext::TexturePool& textures,
		DestructionContext::TextureHandle texture,
		VmaAllocator& allocator,
		std::string_view filename) -> void;
}
