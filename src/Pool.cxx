#include "../include/Pool.hxx"
#include "../include/BindlessHeadless.hxx"
#include "../include/BindlessSet.hxx"


auto DestructionContext::get_device() const -> VkDevice {
    VmaAllocatorInfo info{};
    vmaGetAllocatorInfo(allocator, &info);
    return info.device;
}

auto DestructionContext::create_texture(OffscreenTarget &&target) -> TextureHandle {
    bindless_set->need_repopulate = true;
    return textures.create(std::move(target));
}

auto DestructionContext::create_sampler(VkSampler &&sampler) -> SamplerHandle {
    bindless_set->need_repopulate = true;
    return samplers.create(std::move(sampler));
}


auto DestructionContext::create_sampler(const VkSamplerCreateInfo info, const std::string_view name) -> SamplerHandle {
    bindless_set->need_repopulate = true;

    VkSamplerCreateInfo ci{info};
    ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    ci.pNext = nullptr;
    ci.flags = 0;

    return create_sampler(::create_sampler(allocator, ci, name));
}

auto DestructionContext::create_buffer(Buffer &&buffer) -> BufferHandle {
    bindless_set->need_repopulate = true;
    return buffers.create(std::move(buffer));
}

auto DestructionContext::device_address(BufferHandle handle) -> u64 {
    if (auto *buf = buffers.get(handle)) {
        return buf->device_address;
    }
    return UINT64_MAX;
}


auto destroy(DestructionContext &ctx,
             DestructionContext::TextureHandle handle,
             u64 retire_value) -> void {
    auto impl = ctx.textures.take(handle);
    if (!impl) {
        return;
    }

    ctx.destroy_queue.enqueue(retire_value, [alloc = ctx.allocator, img = std::move(*impl)]() {
        VmaAllocatorInfo info{};
        vmaGetAllocatorInfo(alloc, &info);
        if (img.storage_view) vkDestroyImageView(info.device, img.storage_view, nullptr);
        if (img.sampled_view) vkDestroyImageView(info.device, img.sampled_view, nullptr);
        if (img.image) vmaDestroyImage(alloc, img.image, img.allocation);
    });
}

auto destroy(DestructionContext &ctx,
             DestructionContext::SamplerHandle handle,
             u64 retire_value) -> void {
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

auto destroy(DestructionContext &ctx,
             DestructionContext::BufferHandle handle,
             u64 retire_value) -> void {
    auto impl = ctx.buffers.take(handle);
    if (!impl) {
        return;
    }

    ctx.destroy_queue.enqueue(retire_value, [alloc = ctx.allocator, buf = std::move(*impl)]() {
        VmaAllocatorInfo info{};
        vmaGetAllocatorInfo(alloc, &info);
        if (buf.buffer) vmaDestroyBuffer(alloc, buf.buffer, buf.allocation);
    });
}
