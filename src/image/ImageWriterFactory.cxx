// ImageWriterFactory.cxx
#include "ImageWriter.hxx"
#include <algorithm>
#include <string>

auto make_bmp_writer() -> std::unique_ptr<IImageWriter>;
auto make_png_writer() -> std::unique_ptr<IImageWriter>;

static auto lowercase(std::string_view s) -> std::string
{
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c){ return char(std::tolower(c)); });
    return out;
}

auto make_image_writer_from_filename(std::string_view filename) -> std::unique_ptr<IImageWriter>
{
    auto dot = filename.find_last_of('.');
    if (dot == std::string_view::npos)
        return make_bmp_writer();

    std::string ext = lowercase(filename.substr(dot + 1));
    if (ext == "png")
        return make_png_writer();

    return make_bmp_writer();
}