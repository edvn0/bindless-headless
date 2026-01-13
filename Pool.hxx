#pragma once

#include "Types.hxx"
#include "Forward.hxx"

#include <vector>
#include <cassert>
#include <optional>
#include <functional>
#include <deque>

#include <vma/vk_mem_alloc.h>


struct DeferredDestroyQueue {
	struct Item {
		u64 retire_value;
		std::function<void()> fn;
	};

	std::deque<Item> items;

	auto enqueue(u64 v, std::function<void()> fn) -> void {
		items.push_back({ v, std::move(fn) });
	}

	auto retire(u64 completed) -> void {
		std::erase_if(items, [&](Item const& it) {
			if (it.retire_value > completed) return false;
			it.fn();
			return true;
			});
	}

	auto empty() const -> bool {
		return items.empty();
	}
};



template<typename ObjectType>
class Handle final {
public:
	Handle() = default;

	[[nodiscard]] auto empty() const -> bool {
		return generation == 0u;
	}

	[[nodiscard]] auto valid() const -> bool {
		return generation != 0u;
	}

	[[nodiscard]] auto index() const -> std::uint32_t {
		return index_;
	}

	[[nodiscard]] auto gen() const -> std::uint32_t {
		return generation;
	}

	[[nodiscard]] auto index_as_void() const -> void* {
		return reinterpret_cast<void*>(
			static_cast<std::uintptr_t>(index_));
	}

	[[nodiscard]] auto handle_as_void() const -> void* {
		static_assert(sizeof(void*) >= sizeof(u64));
		auto packed =
			(static_cast<u64>(generation) << 32) |
			static_cast<u64>(index_);
		return reinterpret_cast<void*>(
			static_cast<std::uintptr_t>(packed));
	}

	auto operator==(Handle const& other) const -> bool {
		return index_ == other.index_ && generation == other.generation;
	}

	auto operator!=(Handle const& other) const -> bool {
		return !(*this == other);
	}

	explicit operator bool() const {
		return generation != 0u;
	}

private:
	template<typename ObjectType_, typename ImplObjectType>
	friend class Pool;

	Handle(std::uint32_t index, std::uint32_t gen) noexcept
		: index_(index), generation(gen) {
	}

	std::uint32_t index_ = 0u;
	std::uint32_t generation = 0u;
};

static_assert(sizeof(Handle<class foo>) == sizeof(u64));

template<typename ObjectType, typename ImplObjectType>
class Pool {
	static constexpr std::uint32_t list_end = 0xffffffffu;

	struct PoolEntry {
		PoolEntry() = default;
		explicit PoolEntry(ImplObjectType&& obj) noexcept
			: object(std::move(obj)) {
		}

		ImplObjectType object{};
		std::uint32_t generation = 1u;
		std::uint32_t next_free = list_end;
	};

public:
	[[nodiscard]] auto create(ImplObjectType&& obj) -> Handle<ObjectType> {
		std::uint32_t idx{};
		if (free_list_head != list_end) {
			idx = free_list_head;
			free_list_head = entries[idx].next_free;
			entries[idx].object = std::move(obj);
		}
		else {
			idx = static_cast<std::uint32_t>(entries.size());
			entries.emplace_back(std::move(obj));
		}
		++object_count;
		return Handle<ObjectType>(idx, entries[idx].generation);
	}

	auto destroy(Handle<ObjectType> handle) -> void {
		if (handle.empty()) {
			return;
		}

		assert(object_count > 0u);
		auto const index = handle.index();
		assert(index < entries.size());
		assert(handle.gen() == entries[index].generation);

		entries[index].object = ImplObjectType{};
		++entries[index].generation;
		entries[index].next_free = free_list_head;
		free_list_head = index;
		--object_count;
	}

	[[nodiscard]] auto take(Handle<ObjectType> handle) -> std::optional<ImplObjectType> {
		if (handle.empty()) {
			return std::nullopt;
		}

		auto const index = handle.index();
		assert(index < entries.size());
		assert(handle.gen() == entries[index].generation);

		ImplObjectType obj = std::move(entries[index].object);
		entries[index].object = ImplObjectType{};
		++entries[index].generation;
		entries[index].next_free = free_list_head;
		free_list_head = index;
		--object_count;
		return obj;
	}

	[[nodiscard]] auto get(Handle<ObjectType> handle) const -> ImplObjectType const* {
		if (handle.empty()) {
			return nullptr;
		}

		auto const index = handle.index();
		assert(index < entries.size());
		assert(handle.gen() == entries[index].generation);
		return &entries[index].object;
	}

	[[nodiscard]] auto get(Handle<ObjectType> handle) -> ImplObjectType* {
		if (handle.empty()) {
			return nullptr;
		}

		auto const index = handle.index();
		assert(index < entries.size());
		assert(handle.gen() == entries[index].generation);
		return &entries[index].object;
	}

	[[nodiscard]] auto get_handle(std::uint32_t index) const -> Handle<ObjectType> {
		assert(index < entries.size());
		if (index >= entries.size()) {
			return Handle<ObjectType>{};
		}
		return Handle<ObjectType>(index, entries[index].generation);
	}

	template<typename Fn>
	auto for_each_live(Fn&& fn) -> void {
		for (std::uint32_t i = 0u; i < entries.size(); ++i) {
			auto const& e = entries[i];
			if (e.generation == 0u) {
				continue;
			}
			if (!is_free(i)) {
				fn(get_handle(i), e.object);
			}
		}
	}

	auto clear() -> void {
		entries.clear();
		free_list_head = list_end;
		object_count = 0u;
	}

	[[nodiscard]] auto num_objects() const -> std::uint32_t {
		return object_count;
	}

	[[nodiscard]] auto data() -> std::vector<PoolEntry>& {
		return entries;
	}

	[[nodiscard]] auto data() const -> std::vector<PoolEntry> const& {
		return entries;
	}

private:
	std::vector<PoolEntry> entries{};
	std::uint32_t free_list_head = list_end;
	std::uint32_t object_count = 0u;

	[[nodiscard]] auto is_free(std::uint32_t index) const -> bool {
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

struct DestructionContext {
	VmaAllocator& allocator;
	DeferredDestroyQueue destroy_queue{};
	BindlessSet* bindless_set{ nullptr };

	using TextureHandle = Handle<struct TextureTag>;
	using TexturePool = Pool<TextureTag, OffscreenTarget>;

	TexturePool textures{};
	auto create_texture(OffscreenTarget&& target) -> TextureHandle;

	using SamplerHandle = Handle<struct SamplerTag>;
	using SamplerPool = Pool<SamplerTag, VkSampler>;

	SamplerPool samplers{};
	auto create_sampler(VkSampler&& sampler) -> SamplerHandle;
};

auto destroy(DestructionContext& ctx,
	DestructionContext::TextureHandle handle,
	u64 retire_value = UINT64_MAX) -> void;
auto destroy(DestructionContext& ctx,
	DestructionContext::SamplerHandle handle,
	u64 retire_value = UINT64_MAX) -> void;
