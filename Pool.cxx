#include "Pool.hxx"
#include "BindlessHeadless.hxx"
#include "BindlessSet.hxx"

auto DestructionContext::create_texture(OffscreenTarget&& target) -> TextureHandle {
	bindless_set->need_repopulate = true;
	return textures.create(std::move(target));
}

auto DestructionContext::create_sampler(VkSampler&& sampler) -> SamplerHandle {
	bindless_set->need_repopulate = true;
	return samplers.create(std::move(sampler));
}

auto destroy(DestructionContext& ctx,
	DestructionContext::TextureHandle handle,
	u64 retire_value) -> void
{
	auto impl = ctx.textures.take(handle);
	if (!impl) {
		return;
	}

	ctx.destroy_queue.enqueue(retire_value, [alloc = ctx.allocator, img = std::move(*impl)]() {
		VmaAllocatorInfo info{};
		vmaGetAllocatorInfo(alloc, &info);
		if (img.storage_view)  vkDestroyImageView(info.device, img.storage_view, nullptr);
		if (img.sampled_view)  vkDestroyImageView(info.device, img.sampled_view, nullptr);
		if (img.image) vmaDestroyImage(alloc, img.image, img.allocation);
		});
}

auto destroy(DestructionContext& ctx,
	DestructionContext::SamplerHandle handle,
	u64 retire_value) -> void
{
	auto impl = ctx.samplers.take(handle);
	if (!impl) {
		return;
	}

	ctx.destroy_queue.enqueue(retire_value, [alloc = ctx.allocator, samp = std::move(*impl)]() {
		VmaAllocatorInfo info{};
		vmaGetAllocatorInfo(alloc, &info);
		vkDestroySampler(info.device, samp, nullptr);
		});
}