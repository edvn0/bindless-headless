#pragma once
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "Types.hxx"

enum class PixelLayout {
    Rgb8,
    Rgba8,
};

struct CpuImage {
    u32 width{0};
    u32 height{0};
    PixelLayout layout{PixelLayout::Rgb8};
    u32 stride_bytes{0};
    std::vector<u8> pixels{};
};

class IImageWriter {
public:
    virtual ~IImageWriter() = default;

    virtual auto extension() const -> std::string_view = 0; // "bmp", "png"
    virtual auto write(std::string_view filename, CpuImage const &img) const -> bool = 0;
};

auto make_image_writer_from_filename(std::string_view filename) -> std::unique_ptr<IImageWriter>;
