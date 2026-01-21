#pragma once

#include "Pool.hxx"

#include <vma/vk_mem_alloc.h>

#include <vulkan/vulkan.h>

#include <fstream>
#include <ostream>
#include <string_view>

namespace image_operations {
    struct ImageWriteRequest {
        const OffscreenTarget *texture;
        std::string filename;
    };

    auto write_to_disk(const OffscreenTarget *, VmaAllocator &allocator, std::string_view filename) -> void;
    auto write_batch_to_disk(VmaAllocator &allocator, std::span<const ImageWriteRequest>) -> void;
} // namespace image_operations
