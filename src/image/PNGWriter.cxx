// PngWriter.cxx
#include "ImageWriter.hxx"
#include <png.h>
#include <cstdio>
#include <vector>

#ifdef _MSC_VER
#pragma warning(disable: 4611)
#pragma warning(disable: 4996)
#endif

namespace {

class PngWriter final : public IImageWriter {
public:
    auto extension() const -> std::string_view override { return "png"; }

    auto write(std::string_view filename, CpuImage const& img) const -> bool override
    {
        using FileHandle = std::unique_ptr<FILE, decltype(&fclose)>;
        FileHandle file(fopen(filename.data(), "wb"), &fclose);
        if (!file)
            return false;

        png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
        if (!png_ptr)
            return false;

        png_infop info_ptr = png_create_info_struct(png_ptr);
        if (!info_ptr) {
            png_destroy_write_struct(&png_ptr, nullptr);
            return false;
        }

        if (setjmp(png_jmpbuf(png_ptr))) {
            png_destroy_write_struct(&png_ptr, &info_ptr);
            return false;
        }

        png_init_io(png_ptr, file.get());

        int const color_type = (img.layout == PixelLayout::Rgba8) ? PNG_COLOR_TYPE_RGBA : PNG_COLOR_TYPE_RGB;

        png_set_IHDR(
            png_ptr, info_ptr,
            img.width, img.height,
            8, color_type,
            PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_DEFAULT,
            PNG_FILTER_TYPE_DEFAULT
        );

        png_set_sRGB(png_ptr, info_ptr, PNG_sRGB_INTENT_PERCEPTUAL);

        png_write_info(png_ptr, info_ptr);

        std::vector<png_bytep> rows(img.height);
        for (u32 y = 0; y < img.height; ++y)
            rows[y] = const_cast<png_bytep>(img.pixels.data() + std::size_t(y) * img.stride_bytes);

        png_write_image(png_ptr, rows.data());
        png_write_end(png_ptr, nullptr);

        png_destroy_write_struct(&png_ptr, &info_ptr);
        return true;
    }
};

} // namespace

auto make_png_writer() -> std::unique_ptr<IImageWriter> { return std::make_unique<PngWriter>(); }
