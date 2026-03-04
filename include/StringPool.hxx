#pragma once

#include <atomic>
#include <array>
#include <string>
#include <string_view>
#include <shared_mutex>
#include <unordered_map>

#include "Types.hxx"

class StringPool {
public:
    using Handle = u32;

    static constexpr u32 SLOTS_PER_BLOCK = 1024;
    static constexpr u32 MAX_BLOCKS = 64;

    static auto intern(std::string_view) -> Handle;
    static auto get_view(Handle handle) -> std::string_view;
    static auto sort_key(Handle handle) -> u32;

private:
    struct Block {
        std::array<std::string, SLOTS_PER_BLOCK> storage{};
    };

    struct Instance {
        std::array<std::atomic<Block*>, MAX_BLOCKS> blocks{};
        std::atomic<u32> count{0};

        std::unordered_map<std::string_view, Handle> lookup;
        std::shared_mutex mutex;

        std::vector<Handle> sorted_order;
        std::vector<u32>    rank_of;

        ~Instance();
    };

    static auto get() -> Instance&;
};

struct FlyString {
    StringPool::Handle handle = 0;

    FlyString() = default;
    explicit(false) FlyString(const char* s) : handle(StringPool::intern(s)) {}
    explicit(false) FlyString(const std::string_view sv) : handle(StringPool::intern(sv)) {}
    explicit(false) FlyString(const std::string& s) : handle(StringPool::intern(s)) {}

    [[nodiscard]] auto c_str() const -> const char*;
    [[nodiscard]] auto view() const -> std::string_view;

    explicit(false) operator std::string_view() const;
    auto operator==(const FlyString& other) const -> bool;
    auto operator!=(const FlyString& other) const -> bool;
    auto operator<=>(const FlyString& o) const {
        return StringPool::sort_key(handle) <=> StringPool::sort_key(o.handle);
    }
};