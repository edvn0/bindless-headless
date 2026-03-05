#include "StringPool.hxx"
#include <mutex>

auto StringPool::intern(const std::string_view sv) -> Handle {
    auto &inst = get();

    {
        std::shared_lock lock(inst.mutex);
        if (const auto it = inst.lookup.find(sv); it != inst.lookup.end())
            return it->second;
    }

    std::unique_lock lock(inst.mutex);

    if (const auto it = inst.lookup.find(sv); it != inst.lookup.end())
        return it->second;

    const auto index = inst.count.load(std::memory_order_relaxed);
    const auto block_idx = index / SLOTS_PER_BLOCK;
    const auto local_idx = index % SLOTS_PER_BLOCK;

    if (block_idx >= MAX_BLOCKS)
        std::abort();

    if (inst.blocks[block_idx].load(std::memory_order_acquire) == nullptr)
        inst.blocks[block_idx].store(new Block(), std::memory_order_release);

    auto *const block = inst.blocks[block_idx].load(std::memory_order_relaxed);
    block->storage[local_idx] = std::string(sv);

    const auto id = static_cast<Handle>(index);
    inst.lookup[block->storage[local_idx]] = id;
    inst.count.fetch_add(1, std::memory_order_release);

    const auto insert_pos = static_cast<u32>(
            std::ranges::lower_bound(inst.sorted_order, id,
                                     [](const auto &a, const auto &b) { return get_view(a) < get_view(b); }) -
            inst.sorted_order.begin());

    inst.sorted_order.insert(inst.sorted_order.begin() + insert_pos, id);

    inst.rank_of.resize(inst.sorted_order.size());
    for (u32 rank = insert_pos; rank < static_cast<u32>(inst.sorted_order.size()); ++rank)
        inst.rank_of[inst.sorted_order[rank]] = rank;

    return id;
}

auto StringPool::sort_key(const Handle handle) -> u32 {
    auto &inst = get();
    std::shared_lock lock(inst.mutex);
    return inst.rank_of[handle];
}

auto StringPool::get_view(const Handle handle) -> std::string_view {
    auto &inst = get();

    if (handle >= inst.count.load(std::memory_order_acquire)) {
        std::abort();
    }

    const auto block_idx = handle / SLOTS_PER_BLOCK;
    const auto local_idx = handle % SLOTS_PER_BLOCK;

    return inst.blocks[block_idx].load(std::memory_order_relaxed)->storage[local_idx];
}

auto StringPool::get() -> Instance & {
    static Instance instance;
    return instance;
}

StringPool::Instance::~Instance() {
    for (auto &b: blocks) {
        delete b.load();
    }
}

auto FlyString::c_str() const -> const char * { return StringPool::get_view(handle).data(); }

auto FlyString::view() const -> std::string_view { return StringPool::get_view(handle); }

FlyString::operator std::string_view() const { return view(); }

auto FlyString::operator==(const FlyString &other) const -> bool { return handle == other.handle; }

auto FlyString::operator!=(const FlyString &other) const -> bool { return handle != other.handle; }
