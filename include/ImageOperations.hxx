#pragma once

#include <functional>
#include <span>
#include <string>
#include <string_view>

#include "Forward.hxx"
using VmaAllocator = struct VmaAllocator_T *;

namespace image_operations {
    struct ImageWriteRequest {
        const OffscreenTarget *texture;
        std::string filename;
    };

    using PercentageProgress = float;
    using ProgressFn = std::function<void(float)>;

    auto write_to_disk(const OffscreenTarget *, VmaAllocator &allocator, std::string_view filename) -> void;
    auto write_batch_to_disk(VmaAllocator &allocator, std::span<const ImageWriteRequest>, ProgressFn = {}) -> void;
} // namespace image_operations
