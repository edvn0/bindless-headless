// SceneInspector.cxx
// Usage: scene_inspect <file.scene.bz2> [options]
//
// Options:
//   --verbose      Print per-vertex AABB and full index ranges
//   --textures     Print per-texture KTX2 metadata (format, dims, mips, supercompression)
//   --dump-ktx <dir>  Write raw KTX2 blobs to <dir>/<name>.ktx2 for external inspection
//
#include "Material.hxx"
#include "SceneLoader.hxx"

#include <bit>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <bzlib.h>
#include <ktx.h>
#include <volk.h>

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#define IS_TTY() (_isatty(_fileno(stdout)) != 0)
#else
#include <unistd.h>
#define IS_TTY() (isatty(STDOUT_FILENO) != 0)
#endif

// ---------------------------------------------------------------------------
// Output buffer — everything writes here, flushed to stdout at the end.
// ---------------------------------------------------------------------------
static std::ostringstream g_out;

// ---------------------------------------------------------------------------
// ANSI colour helpers — no-ops when stdout is not a TTY.
// ---------------------------------------------------------------------------
static bool g_color = false;

namespace col {
    static auto reset() -> std::string_view { return g_color ? "\033[0m" : ""; }
    static auto bold() -> std::string_view { return g_color ? "\033[1m" : ""; }
    static auto dim() -> std::string_view { return g_color ? "\033[2m" : ""; }
    static auto cyan() -> std::string_view { return g_color ? "\033[96m" : ""; }
    static auto yellow() -> std::string_view { return g_color ? "\033[93m" : ""; }
    static auto green() -> std::string_view { return g_color ? "\033[92m" : ""; }
    static auto red() -> std::string_view { return g_color ? "\033[91m" : ""; }
    static auto magenta() -> std::string_view { return g_color ? "\033[95m" : ""; }
} // namespace col

// ---------------------------------------------------------------------------
// Tiny helpers
// ---------------------------------------------------------------------------

static constexpr u64 k_file_prefix_magic = 0x454E4543534E5331ULL; // 'SNS1CENE'
static constexpr size_t k_prefix_bytes = 16; // [magic u64][src_hash u64]

static auto read_file(const std::filesystem::path &p) -> std::vector<std::byte> {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        std::cerr << col::red() << "error: cannot open '" << p.string() << "'" << col::reset() << "\n";
        std::exit(1);
    }
    f.seekg(0, std::ios::end);
    const auto sz = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<std::byte> buf(sz);
    f.read(std::bit_cast<char *>(buf.data()), static_cast<std::streamsize>(sz));
    return buf;
}

static auto bzip2_decompress(std::span<const std::byte> src) -> std::vector<std::byte> {
    size_t cap = src.size() * 8;
    for (int attempt = 0; attempt < 7; ++attempt) {
        std::vector<std::byte> dst(cap);
        unsigned int out_len = static_cast<unsigned int>(cap);
        const int rc = BZ2_bzBuffToBuffDecompress(std::bit_cast<char *>(dst.data()), &out_len,
                                                  const_cast<char *>(std::bit_cast<const char *>(src.data())),
                                                  static_cast<unsigned int>(src.size()), 0, 0);
        if (rc == BZ_OK) {
            dst.resize(out_len);
            return dst;
        }
        if (rc != BZ_OUTBUFF_FULL) {
            std::cerr << col::red() << "bzip2 error: " << rc << col::reset() << "\n";
            std::exit(1);
        }
        cap *= 4;
    }
    std::cerr << col::red() << "bzip2: output buffer never large enough\n" << col::reset();
    std::exit(1);
}

template<class T>
static auto blob_span(std::span<const std::byte> file, const Tooling::BlobRange &r) -> std::span<const T> {
    const size_t off = static_cast<size_t>(r.offset);
    const size_t sz = static_cast<size_t>(r.size);
    if (off + sz > file.size())
        return {};
    return {reinterpret_cast<const T *>(file.data() + off), sz / sizeof(T)};
}

static auto string_at(std::span<const std::byte> blob, u32 offset) -> std::string_view {
    if (offset >= blob.size())
        return "<out-of-range>";
    return std::string_view{reinterpret_cast<const char *>(blob.data() + offset)};
}

static auto human_bytes(u64 n) -> std::string {
    if (n < 1024)
        return std::format("{} B", n);
    if (n < 1024 * 1024)
        return std::format("{:.1f} KiB", n / 1024.0);
    if (n < 1024 * 1024 * 1024)
        return std::format("{:.1f} MiB", n / (1024.0 * 1024.0));
    return std::format("{:.2f} GiB", n / (1024.0 * 1024.0 * 1024.0));
}

// ---------------------------------------------------------------------------
// VkFormat name lookup (covers the formats you actually emit)
// ---------------------------------------------------------------------------
static auto vk_format_name(u32 fmt) -> std::string_view {
    switch (fmt) {
        case 37:
            return "VK_FORMAT_R8G8B8A8_UNORM";
        case 43:
            return "VK_FORMAT_R8G8B8A8_SRGB";
        case 131:
            return "VK_FORMAT_BC7_UNORM_BLOCK";
        case 132:
            return "VK_FORMAT_BC7_SRGB_BLOCK";
        case 149:
            return "VK_FORMAT_ASTC_4x4_UNORM_BLOCK";
        case 157:
            return "VK_FORMAT_ASTC_4x4_SRGB_BLOCK";
        default:
            return "VK_FORMAT_UNKNOWN";
    }
}

// ---------------------------------------------------------------------------
// KTX2 metadata printer
// ---------------------------------------------------------------------------
static auto inspect_ktx2(std::span<const std::byte> bytes, std::string_view name, bool verbose) -> void {
    if (!name.empty())
        g_out << "      name       : " << col::bold() << name << col::reset() << "\n";

    ktxTexture2 *ktx2 = nullptr;
    const KTX_error_code rc = ktxTexture_CreateFromMemory(
            reinterpret_cast<const ktx_uint8_t *>(bytes.data()), static_cast<ktx_size_t>(bytes.size()),
            KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, reinterpret_cast<ktxTexture **>(&ktx2));

    if (rc != KTX_SUCCESS || !ktx2) {
        g_out << "      " << col::red() << "[KTX2 parse failed: " << rc << "]" << col::reset() << "\n";
        return;
    }

    const u32 vk_fmt = ktx2->vkFormat;
    g_out << "      format     : " << col::cyan() << vk_format_name(vk_fmt) << col::dim() << " (" << vk_fmt << ")"
          << col::reset() << "\n";
    g_out << "      dims       : " << ktx2->baseWidth << " x " << ktx2->baseHeight;
    if (ktx2->baseDepth > 1)
        g_out << " x " << ktx2->baseDepth;
    g_out << "\n";
    g_out << "      mip levels : " << ktx2->numLevels << "\n";
    g_out << "      layers     : " << ktx2->numLayers << "  faces: " << ktx2->numFaces << "\n";
    g_out << "      is array   : " << (ktx2->isArray ? "yes" : "no") << "\n";

    std::string_view sc_name = "none";
    switch (ktx2->supercompressionScheme) {
        case KTX_SS_NONE:
            sc_name = "none";
            break;
        case KTX_SS_BASIS_LZ:
            sc_name = "BasisLZ";
            break;
        case KTX_SS_ZSTD:
            sc_name = "Zstd";
            break;
        case KTX_SS_ZLIB:
            sc_name = "zlib";
            break;
        default:
            sc_name = "unknown";
            break;
    }
    g_out << "      supercomp  : " << col::yellow() << sc_name << col::reset();
    if (ktxTexture2_NeedsTranscoding(ktx2))
        g_out << " " << col::magenta() << "(needs transcoding)" << col::reset();
    g_out << "\n";

    {
        char *value = nullptr;
        u32 length = 0;
        if (KTX_SUCCESS ==
            ktxHashList_FindValue(&ktx2->kvDataHead, "KTXwriterScParams", &length, reinterpret_cast<void **>(&value)))
            g_out << "      sc params  : " << col::dim() << std::string_view(value, length) << col::reset() << "\n";
    }
    {
        char *value = nullptr;
        u32 length = 0;
        if (KTX_SUCCESS ==
            ktxHashList_FindValue(&ktx2->kvDataHead, "KTXwriter", &length, reinterpret_cast<void **>(&value)))
            g_out << "      writer     : " << col::dim() << std::string_view(value, length) << col::reset() << "\n";
    }

    if (verbose) {
        u64 total_data = 0;
        for (u32 level = 0; level < ktx2->numLevels; ++level) {
            const auto level_sz = ktxTexture_GetImageSize(reinterpret_cast<ktxTexture *>(ktx2), level);
            const u32 mip_w = std::max(1u, ktx2->baseWidth >> level);
            const u32 mip_h = std::max(1u, ktx2->baseHeight >> level);
            g_out << std::format("        mip[{:2d}] {}x{} — {}\n", level, mip_w, mip_h, human_bytes(level_sz));
            total_data += level_sz;
        }
        g_out << "      total data : " << human_bytes(total_data) << "\n";
    }

    ktxTexture_Destroy(reinterpret_cast<ktxTexture *>(ktx2));
}

// ---------------------------------------------------------------------------
// AABB over the vertex buffer
// ---------------------------------------------------------------------------
struct Aabb {
    std::array<float, 3> min_v{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                               std::numeric_limits<float>::max()};
    std::array<float, 3> max_v{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
                               -std::numeric_limits<float>::max()};

    auto extend(const std::array<float, 3> &p) -> void {
        for (int i = 0; i < 3; ++i) {
            min_v[i] = std::min(min_v[i], p[i]);
            max_v[i] = std::max(max_v[i], p[i]);
        }
    }
};

static auto compute_aabb(std::span<const Tooling::Vertex> verts) -> Aabb {
    Aabb a{};
    for (const auto &v: verts)
        a.extend(v.position);
    return a;
}

static constexpr u32 FLAG_ALBEDO_MAP = 1 << 0;
static constexpr u32 FLAG_NORMAL_MAP = 1 << 1;
static constexpr u32 FLAG_ROUGHNESS_MAP = 1 << 2;
static constexpr u32 FLAG_METALLIC_MAP = 1 << 3;
static constexpr u32 FLAG_OCCLUSION_MAP = 1 << 4;
static constexpr u32 FLAG_EMISSIVE_MAP = 1 << 5;
static constexpr u32 FLAG_ALPHA_TESTED = 1 << 6;

static auto decode_material_flags(u32 flags) -> std::string {
    if (flags == 0)
        return "none";
    std::string out;
    auto add = [&](u32 f, std::string_view name) {
        if (flags & f) {
            if (!out.empty())
                out += " | ";
            out += name;
        }
    };
    add(FLAG_ALBEDO_MAP, "albedo_map");
    add(FLAG_NORMAL_MAP, "normal_map");
    add(FLAG_ROUGHNESS_MAP, "roughness_map");
    add(FLAG_METALLIC_MAP, "metallic_map");
    add(FLAG_OCCLUSION_MAP, "occlusion_map");
    add(FLAG_EMISSIVE_MAP, "emissive_map");
    add(FLAG_ALPHA_TESTED, "alpha_tested");
    return out;
}
static auto decode_material_flags(MaterialFlags flags) -> std::string {
    return decode_material_flags(static_cast<u32>(flags));
}

// ---------------------------------------------------------------------------
// Print helpers
// ---------------------------------------------------------------------------
static auto section(std::string_view title) -> void {
    g_out << "\n"
          << col::bold() << col::cyan() << "━━━ " << title << " "
          << std::string(std::max(0, 60 - static_cast<int>(title.size()) - 4), '-') << col::reset() << "\n\n";
}

static auto kv(std::string_view key, auto val, int indent = 0) -> void {
    g_out << std::string(indent, ' ') << col::dim() << key << col::reset() << " : " << val << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    g_color = IS_TTY();

    bool opt_verbose = false;
    bool opt_textures = false;
    bool opt_dump_ktx = false;
    std::string dump_dir = ".";
    std::filesystem::path input_path;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--verbose")
            opt_verbose = true;
        else if (arg == "--textures")
            opt_textures = true;
        else if (arg == "--dump-ktx") {
            opt_dump_ktx = true;
            if (i + 1 < argc)
                dump_dir = argv[++i];
        } else {
            input_path = arg;
        }
    }

    if (input_path.empty()) {
        std::cerr << "Usage: scene_inspect <file.scene.bz2> [--verbose] [--textures] [--dump-ktx <dir>]\n";
        return 1;
    }

    const auto raw = read_file(input_path);
    if (raw.size() <= k_prefix_bytes) {
        std::cerr << col::red() << "File too small\n" << col::reset();
        return 1;
    }

    u64 prefix_magic = 0;
    u64 src_hash = 0;
    std::memcpy(&prefix_magic, raw.data(), 8);
    std::memcpy(&src_hash, raw.data() + 8, 8);

    g_out << col::bold() << col::green() << "\n  scene_inspect — " << input_path.filename().string() << col::reset()
          << "\n";

    section("File");
    kv("path", input_path.string());
    kv("file size", human_bytes(raw.size()));
    kv("prefix magic", std::format("{:#018x}", prefix_magic));
    if (prefix_magic != k_file_prefix_magic)
        g_out << col::red() << "  WARNING: prefix magic mismatch (expected "
              << std::format("{:#018x}", k_file_prefix_magic) << ")\n"
              << col::reset();
    kv("src hash", std::format("{:#018x}", src_hash));

    const std::span<const std::byte> compressed(raw.data() + k_prefix_bytes, raw.size() - k_prefix_bytes);
    std::cout << col::dim() << "  decompressing (" << human_bytes(compressed.size()) << " compressed) …" << col::reset()
              << std::flush;
    const auto decompressed = bzip2_decompress(compressed);
    std::cout << " → " << human_bytes(decompressed.size()) << "\n";

    const std::span<const std::byte> file(decompressed);

    if (file.size() < sizeof(Tooling::FileHeader)) {
        std::cerr << col::red() << "Decompressed data too small for header\n" << col::reset();
        return 1;
    }

    Tooling::FileHeader hdr{};
    std::memcpy(&hdr, file.data(), sizeof(hdr));

    section("Header");
    kv("magic", std::format("{:#010x}  ({})", hdr.magic, (hdr.magic == Tooling::k_magic) ? "OK" : "BAD"));
    kv("version", hdr.version);
    kv("flags", std::format("{:#010x}", hdr.flags));
    kv("content hash", std::format("{:#018x}", hdr.content_hash));
    kv("submesh count", hdr.submesh_count);
    kv("vertex count", hdr.vertex_count);
    kv("index count", hdr.index_count);
    kv("material count", hdr.material_count);
    kv("texture count", hdr.texture_count);

    g_out << "\n  blob layout:\n";
    auto print_blob = [&](std::string_view name, const Tooling::BlobRange &r) {
        g_out << "    " << col::dim() << std::format("{:<20}", name) << col::reset()
              << " offset=" << std::format("{:>10}", r.offset)
              << "  size=" << std::format("{:>12}", human_bytes(r.size)) << "\n";
    };
    print_blob("submesh_table", hdr.submesh_table);
    print_blob("vertex_blob", hdr.vertex_blob);
    print_blob("index_blob", hdr.index_blob);
    print_blob("material_table", hdr.material_table);
    print_blob("texture_table", hdr.texture_table);
    print_blob("string_blob", hdr.string_blob);
    print_blob("texture_blob", hdr.texture_blob);

    const auto file_submeshes = blob_span<Tooling::Submesh>(file, hdr.submesh_table);
    const auto file_vertices = blob_span<Tooling::Vertex>(file, hdr.vertex_blob);
    const auto file_indices = blob_span<u32>(file, hdr.index_blob);
    const auto file_materials = blob_span<Tooling::GPUMaterial>(file, hdr.material_table);
    const auto file_textures = blob_span<Tooling::Texture>(file, hdr.texture_table);
    const auto string_blob =
            file.subspan(static_cast<size_t>(hdr.string_blob.offset), static_cast<size_t>(hdr.string_blob.size));

    section("Geometry");
    kv("vertices", std::format("{} × {} B = {}", file_vertices.size(), sizeof(Tooling::Vertex),
                               human_bytes(file_vertices.size() * sizeof(Tooling::Vertex))));
    kv("indices", std::format("{} ({} triangles) = {}", file_indices.size(), file_indices.size() / 3,
                              human_bytes(file_indices.size() * sizeof(u32))));

    if (!file_vertices.empty()) {
        const Aabb aabb = compute_aabb(file_vertices);
        kv("AABB min", std::format("({:.4f}, {:.4f}, {:.4f})", aabb.min_v[0], aabb.min_v[1], aabb.min_v[2]));
        kv("AABB max", std::format("({:.4f}, {:.4f}, {:.4f})", aabb.max_v[0], aabb.max_v[1], aabb.max_v[2]));
        const float dx = aabb.max_v[0] - aabb.min_v[0];
        const float dy = aabb.max_v[1] - aabb.min_v[1];
        const float dz = aabb.max_v[2] - aabb.min_v[2];
        kv("AABB size", std::format("({:.4f}, {:.4f}, {:.4f})", dx, dy, dz));
    }

    section("Submeshes");
    for (u32 i = 0; i < static_cast<u32>(file_submeshes.size()); ++i) {
        const auto &sm = file_submeshes[i];
        g_out << col::bold() << "  [" << i << "]" << col::reset() << "  vert_offset=" << sm.vertex_offset
              << "  vert_count=" << sm.vertex_count << "  material=" << sm.material_index << "\n";

        for (u32 lod = 0; lod < Tooling::k_lod_count; ++lod) {
            const auto &lr = sm.lods[lod];
            if (lr.index_count == 0) {
                if (opt_verbose)
                    g_out << "    lod[" << lod << "] <not present>\n";
                continue;
            }
            g_out << std::format("    lod[{}]  idx_offset={:>8}  idx_count={:>8}  ({} tris)\n", lod, lr.index_offset,
                                 lr.index_count, lr.index_count / 3);
        }
    }

    section("Materials");
    for (u32 i = 0; i < static_cast<u32>(file_materials.size()); ++i) {
        const auto &m = file_materials[i];
        g_out << col::bold() << "  [" << i << "]" << col::reset() << "\n";

        auto tex_ref = [&](std::string_view label, u32 idx) {
            if (idx == 0xFFFFFFFFu)
                g_out << std::format("    {:<18}: none\n", label);
            else
                g_out << std::format("    {:<18}: tex[{}]\n", label, idx);
        };

        tex_ref("albedo_map", m.albedo_map);
        g_out << std::format("    {:<18}: ({:.3f}, {:.3f}, {:.3f}, {:.3f})\n", "albedo_factor", m.albedo_factor[0],
                             m.albedo_factor[1], m.albedo_factor[2], m.albedo_factor[3]);

        tex_ref("normal_map", m.normal_map);
        tex_ref("roughness_map", m.roughness_map);
        g_out << std::format("    {:<18}: {:.3f}\n", "roughness_factor", m.roughness_factor);

        tex_ref("metallic_map", m.metallic_map);
        g_out << std::format("    {:<18}: {:.3f}\n", "metallic_factor", m.metallic_factor);

        tex_ref("occlusion_map", m.occlusion_map);
        tex_ref("emissive_map", m.emissive_map);
        g_out << std::format("    {:<18}: ({:.3f}, {:.3f}, {:.3f})\n", "emissive_factor", m.emissive_factor[0],
                             m.emissive_factor[1], m.emissive_factor[2]);

        g_out << "    flags             : " << col::yellow() << decode_material_flags(m.flags) << col::reset() << "\n";
    }

    section("Textures");
    for (u32 i = 0; i < static_cast<u32>(file_textures.size()); ++i) {
        const auto &t = file_textures[i];

        const auto name = string_at(string_blob, t.name_str);
        const auto original_path = string_at(string_blob, t.original_path_str);

        g_out << col::bold() << "  [" << i << "] " << col::cyan() << (name.empty() ? "<unnamed>" : name) << col::reset()
              << "\n";
        g_out << "      original path : " << col::dim() << original_path << col::reset() << "\n";
        g_out << "      ktx2 offset   : " << t.ktx2_offset << "  size: " << human_bytes(t.ktx2_size) << "\n";

        const size_t ktx_off = static_cast<size_t>(t.ktx2_offset);
        const size_t ktx_sz = static_cast<size_t>(t.ktx2_size);
        const bool in_range = (ktx_off + ktx_sz <= file.size());

        if (!in_range) {
            g_out << "      " << col::red() << "WARNING: KTX2 data out of range\n" << col::reset();
            continue;
        }

        const std::span<const std::byte> ktx_bytes(file.data() + ktx_off, ktx_sz);

        if (opt_textures || opt_dump_ktx) {
            if (opt_textures)
                inspect_ktx2(ktx_bytes, name, opt_verbose);

            if (opt_dump_ktx) {
                std::filesystem::create_directories(dump_dir);
                const std::string stem = name.empty() ? std::format("texture_{}", i) : std::string(name);
                const auto out_path = std::filesystem::path(dump_dir) / (stem + ".ktx2");
                std::ofstream f(out_path, std::ios::binary);
                f.write(std::bit_cast<const char *>(ktx_bytes.data()), static_cast<std::streamsize>(ktx_bytes.size()));
                g_out << "      " << col::green() << "dumped → " << out_path.string() << col::reset() << "\n";
            }
        }
    }

    section("Summary");
    const u64 geo_bytes = file_vertices.size() * sizeof(Tooling::Vertex) + file_indices.size() * sizeof(u32);
    const u64 mat_bytes = file_materials.size() * sizeof(Tooling::GPUMaterial);
    const u64 tex_bytes = static_cast<u64>(hdr.texture_blob.size);
    const u64 other_bytes = decompressed.size() - geo_bytes - mat_bytes - tex_bytes;

    g_out << "  decompressed total : " << human_bytes(decompressed.size()) << "\n";
    g_out << "    geometry         : " << human_bytes(geo_bytes) << "\n";
    g_out << "    materials        : " << human_bytes(mat_bytes) << "\n";
    g_out << "    KTX2 textures    : " << human_bytes(tex_bytes) << "\n";
    g_out << "    other (hdrs/str) : " << human_bytes(other_bytes) << "\n";
    g_out << "  compressed size    : " << human_bytes(compressed.size())
          << std::format(" ({:.1f}x reduction)\n",
                         static_cast<double>(decompressed.size()) / static_cast<double>(compressed.size()));
    g_out << "\n";

    std::cout << g_out.str() << std::flush;

    return 0;
}
