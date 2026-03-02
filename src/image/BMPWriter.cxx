// BmpWriter.cxx
#include <cstring>
#include <fstream>
#include "ImageWriter.hxx"

namespace {

    auto channel_count(PixelLayout layout) -> u32 { return layout == PixelLayout::Rgba8 ? 4u : 3u; }

    void write_bmp_headers(std::ofstream &output, u32 width, u32 height) {
        const u32 row_size = ((width * 3 + 3) / 4) * 4;
        const u32 pixel_bytes = row_size * height;
        const u32 file_size = 14 + 40 + pixel_bytes;

        const u32 bf_type = 0x4D42;
        const u32 bf_off_bits = 14 + 40;

        output.write(reinterpret_cast<char const *>(&bf_type), 2);
        output.write(reinterpret_cast<char const *>(&file_size), 4);

        u32 reserved = 0;
        output.write(reinterpret_cast<char const *>(&reserved), 4);
        output.write(reinterpret_cast<char const *>(&bf_off_bits), 4);

        u32 bi_size = 40;
        i32 bi_width = static_cast<i32>(width);
        i32 bi_height = static_cast<i32>(height); // positive => bottom-up
        u32 bi_planes = 1;
        u32 bi_bit_count = 24;
        u32 bi_compression = 0;
        u32 bi_size_image = pixel_bytes;
        i32 ppm = 0;
        u32 clr = 0;

        output.write(reinterpret_cast<char const *>(&bi_size), 4);
        output.write(reinterpret_cast<char const *>(&bi_width), 4);
        output.write(reinterpret_cast<char const *>(&bi_height), 4);
        output.write(reinterpret_cast<char const *>(&bi_planes), 2);
        output.write(reinterpret_cast<char const *>(&bi_bit_count), 2);
        output.write(reinterpret_cast<char const *>(&bi_compression), 4);
        output.write(reinterpret_cast<char const *>(&bi_size_image), 4);
        output.write(reinterpret_cast<char const *>(&ppm), 4);
        output.write(reinterpret_cast<char const *>(&ppm), 4);
        output.write(reinterpret_cast<char const *>(&clr), 4);
        output.write(reinterpret_cast<char const *>(&clr), 4);
    }

    class BmpWriter final : public IImageWriter {
    public:
        auto extension() const -> std::string_view override { return "bmp"; }

        auto write(std::string_view filename, CpuImage const &img) const -> bool override {
            std::ofstream out(filename.data(), std::ios::binary);
            if (!out)
                return false;

            write_bmp_headers(out, img.width, img.height);

            u32 const src_channels = channel_count(img.layout);
            u32 const dst_stride = ((img.width * 3 + 3) / 4) * 4;

            std::vector<u8> row(dst_stride);

            for (i32 y = static_cast<i32>(img.height) - 1; y >= 0; --y) {
                u8 const *s = img.pixels.data() + std::size_t(y) * img.stride_bytes;
                u8 *d = row.data();

                for (u32 x = 0; x < img.width; ++x) {
                    d[0] = s[2]; // B
                    d[1] = s[1]; // G
                    d[2] = s[0]; // R
                    s += src_channels;
                    d += 3;
                }

                std::memset(row.data() + img.width * 3, 0, dst_stride - img.width * 3);
                out.write(reinterpret_cast<char const *>(row.data()), row.size());
            }

            return true;
        }
    };

} // namespace

auto make_bmp_writer() -> std::unique_ptr<IImageWriter> { return std::make_unique<BmpWriter>(); }
