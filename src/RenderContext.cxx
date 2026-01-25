#include "../include/BindlessHeadless.hxx"
#include "../include/BindlessSet.hxx"
#include "../include/Pool.hxx"


auto RenderContext::get_device() const -> VkDevice {
    VmaAllocatorInfo info{};
    vmaGetAllocatorInfo(allocator, &info);
    return info.device;
}

auto RenderContext::create_texture(OffscreenTarget &&target) -> TextureHandle {
    bindless_set->need_repopulate = true;
    return textures.create(std::move(target));
}

auto RenderContext::create_sampler(VkSampler &&sampler) -> SamplerHandle {
    bindless_set->need_repopulate = true;
    return samplers.create(std::move(sampler));
}


auto RenderContext::create_sampler(const VkSamplerCreateInfo info, const std::string_view name) -> SamplerHandle {
    bindless_set->need_repopulate = true;

    VkSamplerCreateInfo ci{info};
    ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    ci.pNext = nullptr;
    ci.flags = 0;

    return create_sampler(::create_sampler(allocator, ci, name));
}

auto RenderContext::create_buffer(Buffer &&buffer) -> BufferHandle { return buffers.create(std::move(buffer)); }

auto RenderContext::create_query_pool(QueryPoolState &&state) -> QueryPoolHandle {
    return query_pools.create(std::move(state));
}

auto RenderContext::device_address(BufferHandle handle) -> DeviceAddress {
    if (const auto *buf = buffers.get(handle)) {
        return buf->device_address();
    }
    return DeviceAddress::Invalid;
}

auto RenderContext::clear_all() -> void {
    textures.for_each_live([&ctx = *this](auto h, auto &) { destroy(ctx, h); });
    samplers.for_each_live([&ctx = *this](auto h, auto &) { destroy(ctx, h); });
    buffers.for_each_live([&ctx = *this](auto h, auto &) { destroy(ctx, h); });
    query_pools.for_each_live([&ctx = *this](auto h, auto &) { destroy(ctx, h); });
}

namespace {
    template<std::size_t N>
    auto destroy_unique_image_views(VkDevice device, std::array<VkImageView, N> views) -> void {
        std::ranges::sort(views);
        VkImageView last = VK_NULL_HANDLE;
        for (VkImageView v: views) {
            if (v == VK_NULL_HANDLE) {
                continue;
            }
            if (v == last) {
                continue;
            }

            vkDestroyImageView(device, v, nullptr);
            last = v;
        }
    }
} // namespace

auto destroy(RenderContext &ctx, TextureHandle handle, u64 retire_value) -> void {
    auto impl = ctx.textures.take(handle);
    if (!impl) {
        return;
    }

    ctx.destroy_queue.enqueue(retire_value, [alloc = ctx.allocator, img = std::move(*impl)]() mutable {
        VmaAllocatorInfo info{};
        vmaGetAllocatorInfo(alloc, &info);

        destroy_unique_image_views(info.device, std::array<VkImageView, 3>{
                                                        img.attachment_view,
                                                        img.sampled_view,
                                                        img.storage_view,
                                                });

        if (img.image != VK_NULL_HANDLE) {
            vmaDestroyImage(alloc, img.image, img.allocation);
            img.image = VK_NULL_HANDLE;
            img.allocation = VK_NULL_HANDLE;
        }
    });
}


auto destroy(RenderContext &ctx, SamplerHandle handle, u64 retire_value) -> void {
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

auto destroy(RenderContext &ctx, BufferHandle handle, u64 retire_value) -> void {
    auto impl = ctx.buffers.take(handle);
    if (!impl) {
        return;
    }

    ctx.destroy_queue.enqueue(retire_value, [alloc = ctx.allocator, buf = std::move(*impl)]() {
        VmaAllocatorInfo info{};
        vmaGetAllocatorInfo(alloc, &info);
        if (buf.buffer())
            vmaDestroyBuffer(alloc, buf.buffer(), buf.allocation());
    });
}

auto destroy(RenderContext &ctx, QueryPoolHandle handle, u64 retire_value) -> void {
    auto impl = ctx.query_pools.take(handle);
    if (!impl) {
        return;
    }

    ctx.destroy_queue.enqueue(retire_value, [alloc = ctx.allocator, pool = std::move(*impl)]() {
        VmaAllocatorInfo info{};
        vmaGetAllocatorInfo(alloc, &info);
        vkDestroyQueryPool(info.device, pool.pool, nullptr);
    });
}
