#include "Compression.hxx"
#include "Logger.hxx"

#if defined(SCENE_COMPRESSION_BZIP2)
#include <bzlib.h>
#elif defined(SCENE_COMPRESSION_ZSTD)
#include <zstd.h>
#elif defined(SCENE_COMPRESSION_LZ4)
#include <lz4frame.h>
#include <lz4hc.h>
#else
#error "No scene compression backend defined. Set SCENE_COMPRESSION in CMake."
#endif


#if defined(SCENE_COMPRESSION_BZIP2)

auto bzip2_compress(const std::filesystem::path &out_path_no_normalize, std::span<std::byte> bytes, u64 src_hash)
        -> tl::expected<void, Error> {
    auto out_path = normalize_scene_out_path(out_path_no_normalize);
    info("Writing output scene file: {}", out_path.string());

    auto compressed_size = static_cast<unsigned int>(static_cast<float>(bytes.size()) * 1.01f + 600);
    std::vector<char> compressed(compressed_size);

    const int bz_rc = BZ2_bzBuffToBuffCompress(compressed.data(), &compressed_size, std::bit_cast<char *>(bytes.data()),
                                               static_cast<unsigned int>(bytes.size()),
                                               /*blockSize100k*/ 9,
                                               /*verbosity*/ 0,
                                               /*workFactor*/ 0);

    if (bz_rc != BZ_OK)
        return tl::unexpected(make_error(std::format("BZ2_bzBuffToBuffCompress failed: {}", bz_rc)));

    std::ofstream f(out_path, std::ios::binary);
    if (!f)
        return tl::unexpected(make_error("Failed to open file for writing: " + out_path.string()));

    f.write(std::bit_cast<const char *>(&k_prefix_magic), 8);
    f.write(std::bit_cast<const char *>(&src_hash), 8);
    f.write(compressed.data(), static_cast<std::streamsize>(compressed_size));

    if (!f)
        return tl::unexpected(make_error("Failed to write file: " + out_path.string()));

    return {};
}

auto bzip2_decompress(std::span<const std::byte> src) -> tl::expected<std::vector<std::byte>, Error> {
    size_t dst_cap = src.size() * 8;
    for (int attempt = 0; attempt < 6; ++attempt) {
        std::vector<std::byte> dst(dst_cap);
        unsigned int dst_len = static_cast<unsigned int>(dst_cap);

        const int rc = BZ2_bzBuffToBuffDecompress(std::bit_cast<char *>(dst.data()), &dst_len,
                                                  const_cast<char *>(std::bit_cast<const char *>(src.data())),
                                                  static_cast<unsigned int>(src.size()),
                                                  /*small*/ 0, /*verbosity*/ 0);

        if (rc == BZ_OK) {
            dst.resize(dst_len);
            return dst;
        }
        if (rc == BZ_OUTBUFF_FULL) {
            dst_cap *= 4;
            continue;
        }
        return tl::unexpected(
                Error::make_error(Error::Type::MeshLoadError, "bzip2 decompress failed: " + std::to_string(rc)));
    }
    return tl::unexpected(Error::make_error(Error::Type::MeshLoadError, "bzip2 output buffer never large enough"));
}
#elif defined(SCENE_COMPRESSION_ZSTD)


auto zstd_compress(const std::filesystem::path &out_path_no_normalize, std::span<std::byte> bytes, u64 src_hash)
        -> tl::expected<void, Error> {
    auto out_path = normalize_scene_out_path(out_path_no_normalize);
    info("Writing output scene file: {}", out_path.string());

    const size_t bound = ZSTD_compressBound(bytes.size());
    std::vector<char> compressed(bound);

    // Level 3 is the zstd sweet spot: near-bzip2 ratio, ~10x faster compress,
    // ~25x faster decompress. Bump to 6 if you want smaller files at cost of
    // slower conversion (still decompresses at the same speed).
    const size_t compressed_size = ZSTD_compress(compressed.data(), bound, bytes.data(), bytes.size(),
                                                 /*level=*/3);

    if (ZSTD_isError(compressed_size))
        return tl::unexpected(make_error(std::format("ZSTD_compress failed: {}", ZSTD_getErrorName(compressed_size))));

    std::ofstream f(out_path, std::ios::binary);
    if (!f)
        return tl::unexpected(make_error("Failed to open file for writing: " + out_path.string()));

    f.write(std::bit_cast<const char *>(&k_prefix_magic), 8);
    f.write(std::bit_cast<const char *>(&src_hash), 8);
    f.write(compressed.data(), static_cast<std::streamsize>(compressed_size));

    if (!f)
        return tl::unexpected(make_error("Failed to write file: " + out_path.string()));

    return {};
}

auto zstd_decompress(std::span<const std::byte> src) -> tl::expected<std::vector<std::byte>, Error> {
    const auto content_size = ZSTD_getFrameContentSize(src.data(), src.size());
    if (content_size == ZSTD_CONTENTSIZE_ERROR)
        return tl::unexpected(Error::make_error(Error::Type::MeshLoadError, "zstd: not a valid zstd frame"));
    if (content_size == ZSTD_CONTENTSIZE_UNKNOWN)
        return tl::unexpected(
                Error::make_error(Error::Type::MeshLoadError, "zstd: content size unknown (streaming frame?)"));

    std::vector<std::byte> dst(static_cast<size_t>(content_size));

    const size_t result = ZSTD_decompress(dst.data(), dst.size(), src.data(), src.size());

    if (ZSTD_isError(result))
        return tl::unexpected(Error::make_error(Error::Type::MeshLoadError,
                                                std::format("zstd decompress failed: {}", ZSTD_getErrorName(result))));

    return dst;
}

#elif defined(SCENE_COMPRESSION_LZ4)

auto lz4_compress(const std::filesystem::path &out_path_no_normalize, std::span<std::byte> bytes, u64 src_hash)
        -> tl::expected<void, Error> {
    auto out_path = normalize_scene_out_path(out_path_no_normalize);
    info("Writing output scene file: {}", out_path.string());

    LZ4F_preferences_t prefs = LZ4F_INIT_PREFERENCES;
    prefs.frameInfo.contentSize = bytes.size();
    prefs.compressionLevel = LZ4HC_CLEVEL_DEFAULT;

    const size_t bound = LZ4F_compressFrameBound(bytes.size(), &prefs);
    std::vector<char> compressed(bound);

    const size_t compressed_size = LZ4F_compressFrame(compressed.data(), bound, bytes.data(), bytes.size(), &prefs);

    if (LZ4F_isError(compressed_size))
        return tl::unexpected(
                make_error(std::format("LZ4F_compressFrame failed: {}", LZ4F_getErrorName(compressed_size))));

    std::ofstream f(out_path, std::ios::binary);
    if (!f)
        return tl::unexpected(make_error("Failed to open file for writing: " + out_path.string()));

    f.write(std::bit_cast<const char *>(&k_prefix_magic), 8);
    f.write(std::bit_cast<const char *>(&src_hash), 8);
    f.write(compressed.data(), static_cast<std::streamsize>(compressed_size));

    if (!f)
        return tl::unexpected(make_error("Failed to write file: " + out_path.string()));

    return {};
}


auto lz4_decompress(std::span<const std::byte> src) -> tl::expected<std::vector<std::byte>, Error> {
    LZ4F_dctx *ctx = nullptr;
    LZ4F_errorCode_t rc = LZ4F_createDecompressionContext(&ctx, LZ4F_VERSION);
    if (LZ4F_isError(rc))
        return tl::unexpected(Error::make_error(Error::Type::MeshLoadError,
                                                std::format("lz4: failed to create ctx: {}", LZ4F_getErrorName(rc))));

    LZ4F_frameInfo_t info = LZ4F_INIT_FRAMEINFO;
    size_t consumed = src.size();
    rc = LZ4F_getFrameInfo(ctx, &info, src.data(), &consumed);
    if (LZ4F_isError(rc)) {
        LZ4F_freeDecompressionContext(ctx);
        return tl::unexpected(Error::make_error(Error::Type::MeshLoadError,
                                                std::format("lz4: bad frame: {}", LZ4F_getErrorName(rc))));
    }

    if (info.contentSize == 0) {
        LZ4F_freeDecompressionContext(ctx);
        return tl::unexpected(Error::make_error(Error::Type::MeshLoadError,
                                                "lz4: content size unknown (was frame compressed without it?)"));
    }

    std::vector<std::byte> dst(static_cast<size_t>(info.contentSize));
    size_t dst_size = dst.size();
    size_t src_remaining = src.size() - consumed;

    rc = LZ4F_decompress(ctx, dst.data(), &dst_size, static_cast<const std::byte *>(src.data()) + consumed,
                         &src_remaining, nullptr);

    LZ4F_freeDecompressionContext(ctx);

    if (LZ4F_isError(rc))
        return tl::unexpected(Error::make_error(Error::Type::MeshLoadError,
                                                std::format("lz4 decompress failed: {}", LZ4F_getErrorName(rc))));

    dst.resize(dst_size);
    return dst;
}
#endif

auto scene_compress(const std::filesystem::path &out_path, std::span<std::byte> bytes, u64 src_hash)
        -> tl::expected<void, Error> {
#if defined(SCENE_COMPRESSION_BZIP2)
    return bzip2_compress(out_path, bytes, src_hash);
#elif defined(SCENE_COMPRESSION_ZSTD)
    return zstd_compress(out_path, bytes, src_hash);
#elif defined(SCENE_COMPRESSION_LZ4)
    return lz4_compress(out_path, bytes, src_hash);
#endif
}

auto scene_decompress(std::span<const std::byte> src) -> tl::expected<std::vector<std::byte>, Error> {
#if defined(SCENE_COMPRESSION_BZIP2)
    return bzip2_decompress(src);
#elif defined(SCENE_COMPRESSION_ZSTD)
    return zstd_decompress(src);
#elif defined(SCENE_COMPRESSION_LZ4)
    return lz4_decompress(src);
#endif
}
