#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>
#include <fstream>
#include <vulkan/vulkan.h>

inline auto pipeline_cache_path(std::int32_t argc, char **argv)
    -> std::optional<std::filesystem::path> {
    if (argc > 1) return std::filesystem::path{argv[1]};

    char *buf{};
    size_t sz{};
    if (const auto ok = _dupenv_s(&buf, &sz, "BH_PIPE_CACHE_PATH") == 0 && buf; !ok) return std::nullopt;

    auto p = std::filesystem::path{buf};
    free(buf);
    return p;
}

struct PipelineCache {
    VkDevice device{VK_NULL_HANDLE};
    VkPipelineCache cache{VK_NULL_HANDLE};
    std::filesystem::path cache_path{};
    bool has_path{false};

    PipelineCache(VkDevice d, std::optional<std::filesystem::path> opt)
        : device{d} {
        if (opt && !opt->empty()) {
            cache_path = std::move(*opt);
            cache_path /= "bindless-headless.cache";
            has_path = true;
        }

        auto initial_data = read_cache();
        std::span<const std::uint8_t> s{initial_data};

        VkPipelineCacheCreateInfo ci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0U,
            .initialDataSize = s.size_bytes(),
            .pInitialData = s.empty() ? nullptr : s.data()
        };

        vk_check(vkCreatePipelineCache(device, &ci, nullptr, &cache));
    }

    PipelineCache(PipelineCache const &) = delete;

    auto operator=(PipelineCache const &) -> PipelineCache & = delete;

    PipelineCache(PipelineCache &&o) noexcept
        : device{o.device},
          cache{o.cache},
          cache_path{std::move(o.cache_path)},
          has_path{o.has_path} {
        o.device = VK_NULL_HANDLE;
        o.cache = VK_NULL_HANDLE;
        o.has_path = false;
    }

    auto operator=(PipelineCache &&o) noexcept -> PipelineCache & {
        if (this == &o) return *this;
        destroy();
        device = o.device;
        cache = o.cache;
        cache_path = std::move(o.cache_path);
        has_path = o.has_path;
        o.device = VK_NULL_HANDLE;
        o.cache = VK_NULL_HANDLE;
        o.has_path = false;
        return *this;
    }

    ~PipelineCache() noexcept { destroy(); }

    auto get() const noexcept -> VkPipelineCache { return cache; }
    operator VkPipelineCache() const noexcept { return cache; }

private:
    auto read_cache() -> std::vector<std::uint8_t> {
        if (!has_path) return {};

        std::error_code ec{};
        if (!std::filesystem::exists(cache_path, ec)) return {};

        auto sz = std::filesystem::file_size(cache_path, ec);
        if (ec || sz == 0U) return {};

        std::vector<std::uint8_t> buf(sz);
        std::ifstream f{cache_path, std::ios::binary};
        if (!f) return {};

        f.read(reinterpret_cast<char *>(buf.data()),
               static_cast<std::streamsize>(buf.size()));
        if (!f) return {};

        return buf;
    }

    auto destroy() noexcept -> void {
        if (device == VK_NULL_HANDLE || cache == VK_NULL_HANDLE) return;

        if (has_path) write_cache();
        vkDestroyPipelineCache(device, cache, nullptr);
        cache = VK_NULL_HANDLE;
    }

    auto write_cache() -> void {
        std::size_t sz{};
        auto r = vkGetPipelineCacheData(device, cache, &sz, nullptr);
        if (r != VK_SUCCESS || sz == 0U) return;

        std::vector<std::uint8_t> buf(sz);
        r = vkGetPipelineCacheData(device, cache, &sz, buf.data());
        if (r != VK_SUCCESS) return;

        std::ofstream f{
            cache_path,
            std::ios::binary | std::ios::out | std::ios::trunc
        };
        if (!f) return;

        f.write(reinterpret_cast<const char *>(buf.data()),
                static_cast<std::streamsize>(sz));
    }
};
