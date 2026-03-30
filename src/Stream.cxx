#include "Stream.hxx"
#include "Compression.hxx"

TeeStreambuf::TeeStreambuf(usize count) : write_time("TeeStreambuf - write to several destinations") {
    sinks.reserve(count);
}

auto TeeStreambuf::overflow(int c) -> int {
    if (c == EOF)
        return !EOF;
    for (auto *sink: sinks) {
        if (sink->sputc(static_cast<char>(c)) == EOF)
            return EOF;
    }
    return c;
}

auto TeeStreambuf::xsputn(const char *data, std::streamsize n) -> std::streamsize {
    for (auto *sink: sinks) {
        if (sink->sputn(data, n) != n)
            return 0;
    }
    return n;
}

auto TeeStreambuf::sync() -> int {
    int result = 0;
    for (auto *sink: sinks)
        if (sink->pubsync() != 0)
            result = -1;
    return result;
}

auto write_scene_multi(const std::vector<std::byte> &data, const std::vector<std::filesystem::path> &destinations)
        -> bool {
    std::vector<std::unique_ptr<std::ofstream>> files;
    files.reserve(destinations.size());

    TeeStreambuf tee{files.size()};
    for (const auto &dest: destinations) {
        std::filesystem::create_directories(dest.parent_path());

        const auto final_output = normalize_scene_out_path(dest);

        auto f = std::make_unique<std::ofstream>(final_output, std::ios::binary);
        if (!f->is_open())
            return false;

        tee.emplace(f->rdbuf()); // Pointer to rdbuf inside the heap-allocated object
        files.push_back(std::move(f));
    }

    std::ostream out{&tee};
    out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));

    return out.good();
}
