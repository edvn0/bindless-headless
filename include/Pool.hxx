#pragma once

#include "Assert.hxx"
#include "Buffer.hxx"
#include "Forward.hxx"
#include "Material.hxx"
#include "Pipelines.hxx"
#include "Types.hxx"

#include <deque>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include <vk_mem_alloc.h>

struct DeferredDestroyQueue {
    struct Item {
        u64 retire_value;
        std::function<void()> fn;
    };

    std::deque<Item> items;

    auto enqueue(u64 v, std::function<void()> fn) -> void {
        items.push_back({
                .retire_value = v,
                .fn = std::move(fn),
        });
    }

    auto retire(u64 completed) -> void {
        std::erase_if(items, [&](Item const &it) {
            if (it.retire_value > completed)
                return false;
            it.fn();
            return true;
        });
    }

    [[nodiscard]] auto empty() const -> bool { return items.empty(); }
};


template<typename>
class Handle final {
public:
    Handle() = default;

    [[nodiscard]] auto empty() const -> bool { return generation == 0u; }

    [[nodiscard]] auto valid() const -> bool { return generation != 0u; }

    [[nodiscard]] auto index() const -> u32 { return index_; }

    [[nodiscard]] auto gen() const -> u32 { return generation; }

    [[nodiscard]] auto index_as_void() const -> void * {
        return std::bit_cast<void *>(static_cast<std::uintptr_t>(index_));
    }

    [[nodiscard]] auto handle_as_void() const -> void * {
        static_assert(sizeof(void *) >= sizeof(u64));
        auto packed = (static_cast<u64>(generation) << 32) | static_cast<u64>(index_);
        return std::bit_cast<void *>(static_cast<std::uintptr_t>(packed));
    }

    auto operator==(Handle const &other) const -> bool {
        return index_ == other.index_ && generation == other.generation;
    }

    auto operator!=(Handle const &other) const -> bool { return !(*this == other); }

    explicit operator bool() const { return generation != 0u; }

private:
    template<typename ObjectType_, typename ImplObjectType>
    friend class Pool;

    Handle(u32 index, u32 gen) noexcept : index_(index), generation(gen) {}

    u32 index_ = 0u;
    u32 generation = 0u;
};
static_assert(std::is_trivially_copyable_v<Handle<class DebugFoo>>);
static_assert(sizeof(Handle<class DebugFoo>) == sizeof(u64));


template<class T>
concept HandleTag = requires { typename Handle<T>; } && std::is_trivially_copyable_v<Handle<T>> &&
                    (sizeof(Handle<T>) == sizeof(u64));
template<HandleTag T>
class Holder final {
    RenderContext *context{nullptr};
    T handle{};

public:
    explicit Holder(RenderContext &ctx, T h) : context(&ctx), handle(h) {}
    explicit Holder(RenderContext *ctx, T h) : context(ctx), handle(h) {
        ASSERT(context != nullptr, "Context was null for some reason?");
    }
    ~Holder() {
        if (context)
            destroy(*context, handle);
    }
    Holder() = default;

    Holder(Holder &&other) noexcept :
        context(std::exchange(other.context, nullptr)), handle(std::exchange(other.handle, {})) {}
    auto operator=(const Holder &) -> Holder & = delete;
    auto operator=(Holder &&other) -> Holder & {
        std::swap(context, other.context);
        std::swap(handle, other.handle);
        return *this;
    }
    auto operator=(std::nullptr_t) -> Holder & {
        reset();
        return *this;
    }
    explicit(false) operator T() const { return handle; }
    auto valid() const { return handle.valid(); }
    auto empty() const { return handle.empty(); }
    void reset() {
        destroy(*context, handle);
        context = nullptr;
        handle = T{};
    }
    auto release() -> T {
        context = nullptr;
        return std::exchange(handle, T{});
    }
    auto index() const { return handle.index(); }
};

template<typename ObjectType, typename ImplObjectType>
class Pool {
    static constexpr u32 list_end = 0xffffffffu;

    struct PoolEntry {
        PoolEntry() = default;

        explicit PoolEntry(ImplObjectType &&obj) noexcept : object(std::move(obj)) {}

        ImplObjectType object{};
        u32 generation = 1u;
        u32 next_free = list_end;
        bool live{false};
    };

public:
    [[nodiscard]] auto create(ImplObjectType &&obj) -> Handle<ObjectType> {
        u32 idx{};
        if (free_list_head != list_end) {
            idx = free_list_head;
            free_list_head = entries[idx].next_free;
            entries[idx].object = std::move(obj);
            entries[idx].live = true;
        } else {
            idx = static_cast<u32>(entries.size());
            auto &object = entries.emplace_back(std::move(obj));
            object.live = true;
        }
        ++object_count;
        return Handle<ObjectType>(idx, entries[idx].generation);
    }

    auto destroy(Handle<ObjectType> handle) -> void {
        if (handle.empty()) {
            return;
        }

        ASSERT(object_count > 0u, "Trying to destroy object from pool that has no live objects");
        auto const index = handle.index();
        ASSERT(index < entries.size(), "Trying to destroy object with out-of-range index");
        ASSERT(handle.gen() == entries[index].generation, "Trying to destroy object with mismatched generation");

        entries[index].object = ImplObjectType{};
        entries[index].live = false;
        entries[index].generation++;
        entries[index].next_free = free_list_head;
        free_list_head = index;
        object_count--;
    }

    [[nodiscard]] auto get(Handle<ObjectType> handle) const -> ImplObjectType const * {
        if (handle.empty()) {
            return nullptr;
        }
        auto const index = handle.index();
        ASSERT(index < entries.size(), "Trying to access object with out-of-range index");

        if (!entries.at(index).live)
            return nullptr;

        ASSERT(index < entries.size(), "Trying to access object with out-of-range index");
        ASSERT(handle.gen() == entries[index].generation, "Trying to access object with mismatched generation");
        return &entries[index].object;
    }

    [[nodiscard]] auto get_multiple(auto &&...handles) const { return std::make_tuple(get(handles)...); }
    [[nodiscard]] auto get_multiple(auto &&...handles) { return std::make_tuple(get(handles)...); }

    [[nodiscard]] auto get(Handle<ObjectType> handle) -> ImplObjectType * {
        if (handle.empty()) {
            return nullptr;
        }
        auto const index = handle.index();
        ASSERT(index < entries.size(), "Trying to access object with out-of-range index");

        if (!entries.at(index).live)
            return nullptr;

        ASSERT(handle.gen() == entries[index].generation, "Trying to access object with mismatched generation");
        return &entries[index].object;
    }

    [[nodiscard]] auto maybe_get_handle(u32 index) const -> Handle<ObjectType> {
        if (index >= entries.size()) {
            return Handle<ObjectType>{};
        }
        return Handle<ObjectType>(index, entries[index].generation);
    }
    [[nodiscard]] auto get_handle(u32 index) const -> Handle<ObjectType> {
        ASSERT(index < entries.size(), "Trying to get handle with out-of-range index");
        return maybe_get_handle(index);
    }


    template<typename Fn>
    auto for_each_live(Fn &&fn) -> void {
        for (u32 i = 0u; i < entries.size(); ++i) {
            if (!entries[i].live)
                continue;
            fn(get_handle(i), entries[i].object);
        }
    }
    template<typename Fn>
    auto for_each_live_with_skip(Fn &&fn, u32 skip = 3U) -> void {
        for (u32 i = skip; i < entries.size(); ++i) {
            if (!entries[i].live)
                continue;
            fn(get_handle(i), entries[i].object);
        }
    }

    auto clear() -> void {
        entries.clear();
        free_list_head = list_end;
        object_count = 0u;
    }

    [[nodiscard]] auto num_objects() const -> u32 { return object_count; }

    [[nodiscard]] auto data() const -> std::span<const PoolEntry> { return std::span<const PoolEntry>{entries}; }

private:
    std::vector<PoolEntry> entries{};
    u32 free_list_head = list_end;
    u32 object_count = 0u;

    [[nodiscard]] auto is_free(u32 index) const -> bool {
        auto cur = free_list_head;
        while (cur != list_end) {
            if (cur == index) {
                return true;
            }
            cur = entries[cur].next_free;
        }
        return false;
    }
};


struct QueryPoolState {
    VkQueryPool pool = VK_NULL_HANDLE;
    u32 query_count = 0;
    double timestamp_period_ns = 1.0; // from VkPhysicalDeviceLimits::timestampPeriod
};


using TextureHandle = Handle<struct TextureTag>;
using TexturePool = Pool<TextureTag, OffscreenTarget>;
using SamplerHandle = Handle<struct SamplerTag>;
using SamplerPool = Pool<SamplerTag, VkSampler>;
using ComparisonSamplerHandle = Handle<struct ComparisonSamplerTag>;
using ComparisonSamplerPool = Pool<ComparisonSamplerTag, VkSampler>;
using BufferHandle = Handle<struct BufferTag>;
using BufferPool = Pool<BufferTag, Buffer>;
using QueryPoolHandle = Handle<struct QueryPoolTag>;
using QueryPoolPool = Pool<QueryPoolTag, QueryPoolState>;
using PipelineHandle = Handle<struct PipelineTag>;
using PipelinePool = Pool<PipelineTag, CompiledPipeline>;
using ShaderHandle = Handle<struct ShaderTag>;
using ShaderPool = Pool<ShaderTag, VkShaderModule>;
using MaterialHandle = Handle<struct MaterialTag>;
struct GPUMaterialPool {
    Pool<struct MaterialTag, GPUMaterialData> cpu_pool{};

    BufferHandle gpu_buffer{};
    bool dirty{true};

    std::vector<u32> pool_to_gpu{};
    std::vector<u32> gpu_to_pool{};
    u32 gpu_count{0};

    auto register_batch(std::span<const GPUMaterialData>) -> u32;
};
auto flush_material_pool(RenderContext &) -> void;


constexpr auto hot_swap = []<typename Handle, typename Value>(Handle &current, Value &&next, RenderContext &ctx,
                                                              u64 retire_val = std::numeric_limits<u64>::max()) {
    Handle old = current;
    current = create(ctx, std::move(next));
    destroy(ctx, old, retire_val);
};
