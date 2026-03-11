#include <tl/expected.hpp>
#include "BindlessHeadless.hxx"
#include "BindlessSet.hxx"
#include "Pool.hxx"


auto RenderContext::get_device() const -> VkDevice {
    VmaAllocatorInfo info{};
    vmaGetAllocatorInfo(allocator, &info);
    return info.device;
}

auto RenderContext::get_physical_device() const -> VkPhysicalDevice {
    VmaAllocatorInfo info{};
    vmaGetAllocatorInfo(allocator, &info);
    return info.physicalDevice;
}

auto RenderContext::get_instance() const -> VkInstance {
    VmaAllocatorInfo info{};
    vmaGetAllocatorInfo(allocator, &info);
    return info.instance;
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

auto RenderContext::create_comparison_sampler(VkSampler &&sampler) -> SamplerHandle {
    bindless_set->need_repopulate = true;
    return comparison_samplers.create(std::move(sampler));
}

auto RenderContext::create_comparison_sampler(const VkSamplerCreateInfo info, const std::string_view name)
        -> SamplerHandle {
    bindless_set->need_repopulate = true;

    VkSamplerCreateInfo ci{info};
    ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    ci.pNext = nullptr;
    ci.flags = 0;

    return create_comparison_sampler(::create_sampler(allocator, ci, name));
}

auto RenderContext::create_buffer(Buffer &&buffer) -> BufferHandle { return buffers.create(std::move(buffer)); }

auto RenderContext::create_query_pool(QueryPoolState &&state) -> QueryPoolHandle {
    return query_pools.create(std::move(state));
}

auto RenderContext::create_pipeline(CompiledPipeline &&state) -> PipelineHandle {
    return pipeline_pool.create(std::move(state));
}

auto RenderContext::create_shader(VkShaderModule &&module) -> ShaderHandle { return shaders.create(std::move(module)); }

auto RenderContext::create_material(GPUMaterialData &&mat) -> MaterialHandle {
    auto handle = materials.cpu_pool.create(std::move(mat));

    constexpr auto max = std::numeric_limits<u32>::max();

    const u32 gpu_slot = materials.gpu_count++;
    if (materials.pool_to_gpu.size() <= handle.index())
        materials.pool_to_gpu.resize(handle.index() + 1, max);
    if (materials.gpu_to_pool.size() <= gpu_slot)
        materials.gpu_to_pool.resize(gpu_slot + 1, max);

    materials.pool_to_gpu[handle.index()] = gpu_slot;
    materials.gpu_to_pool[gpu_slot] = handle.index();
    materials.dirty = true;
    return handle;
}

auto RenderContext::device_address(BufferHandle handle) -> DeviceAddress {
    if (const auto *buf = buffers.get(handle)) [[likely]] {
        return buf->device_address();
    }
    return DeviceAddress::Invalid;
}
auto RenderContext::device_address(const BufferHandle handle) const -> DeviceAddress {
    if (const auto *buf = buffers.get(handle)) [[likely]] {
        return buf->device_address();
    }
    return DeviceAddress::Invalid;
}
auto RenderContext::get_mapped_pointer(const BufferHandle handle) const -> tl::expected<void *, Error> {
    if (const auto *buf = buffers.get(handle)) [[likely]] {
        vmaInvalidateAllocation(allocator, buf->allocation(), 0, VK_WHOLE_SIZE);
        return buf->data_pointer();
    }
    return tl::make_unexpected(
            Error::make_error(Error::Type::CouldNotMapMemory, "Buffer could not or was not mapped."));
}
auto RenderContext::flush_mapped_memory(const BufferHandle handle, std::size_t offset, std::size_t size) const -> void {
    if (const auto *buf = buffers.get(handle)) [[likely]] {
        vmaFlushAllocation(allocator, buf->allocation(), offset, size);
    }
}
auto RenderContext::texture_format(TextureHandle handle) const -> VkFormat {
    if (const auto *target = textures.get(handle)) [[likely]] {
        return target->format;
    }
    return VK_FORMAT_UNDEFINED;
}

auto RenderContext::clear_all() -> void {
    textures.for_each_live([&ctx = *this](auto h, auto &) { destroy(ctx, h); });
    samplers.for_each_live([&ctx = *this](auto h, auto &) { destroy(ctx, h); });
    comparison_samplers.for_each_live([&ctx = *this](auto h, auto &) { destroy(ctx, h); });
    buffers.for_each_live([&ctx = *this](auto h, auto &) { destroy(ctx, h); });
    query_pools.for_each_live([&ctx = *this](auto h, auto &) { destroy(ctx, h); });
    pipeline_pool.for_each_live([&ctx = *this](auto h, auto &) { destroy(ctx, h); });
    shaders.for_each_live([&ctx = *this](auto h, auto &) { destroy(ctx, h); });
    pipeline_cache.reset();
}

namespace {
    template<std::size_t N>
    auto destroy_unique_image_views(VkDevice device, std::array<VkImageView, N> &&views) -> void {
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
    auto impl = ctx.textures.get(handle);
    if (!impl) {
        return;
    }
    ctx.bindless_set->need_repopulate = true;

    // Extract only the Vulkan handles needed for cleanup
    VkImage image = impl->image;
    VmaAllocation allocation = impl->allocation;
    VkImageView attachment_view = impl->attachment_view;
    VkImageView sampled_view = impl->sampled_view;
    VkImageView storage_view = impl->storage_view;

    ctx.destroy_queue.enqueue(
            retire_value, [alloc = ctx.allocator, image, allocation, attachment_view, sampled_view, storage_view]() {
                VmaAllocatorInfo info{};
                vmaGetAllocatorInfo(alloc, &info);
                destroy_unique_image_views(info.device, std::array<VkImageView, 3>{
                                                                attachment_view,
                                                                sampled_view,
                                                                storage_view,
                                                        });
                if (image != VK_NULL_HANDLE) {
                    vmaDestroyImage(alloc, image, allocation);
                }
            });
    ctx.textures.destroy(handle);
}

auto destroy(RenderContext &ctx, SamplerHandle handle, u64 retire_value) -> void {
    if (handle.empty())
        return;

    VkSampler sampler_to_destroy = VK_NULL_HANDLE;
    bool is_comparison = false;

    if (ctx.comparison_samplers.maybe_get_handle(handle.index()) == handle) {
        sampler_to_destroy = *ctx.comparison_samplers.get(handle);
        is_comparison = true;
    } else if (ctx.samplers.maybe_get_handle(handle.index()) == handle) {
        sampler_to_destroy = *ctx.samplers.get(handle);
        is_comparison = false;
    }

    if (sampler_to_destroy == VK_NULL_HANDLE) {
        return;
    }

    ctx.bindless_set->need_repopulate = true;

    ctx.destroy_queue.enqueue(retire_value, [device = ctx.get_device(), sampler = sampler_to_destroy]() {
        vkDestroySampler(device, sampler, nullptr);
    });

    if (is_comparison) {
        ctx.comparison_samplers.destroy(handle);
    } else {
        ctx.samplers.destroy(handle);
    }
}

auto destroy(RenderContext &ctx, BufferHandle handle, u64 retire_value) -> void {
    auto impl = ctx.buffers.get(handle);
    if (!impl) {
        return;
    }

    // Extract buffer and allocation handles
    VkBuffer buffer = impl->buffer();
    VmaAllocation allocation = impl->allocation();

    ctx.destroy_queue.enqueue(retire_value, [alloc = ctx.allocator, buffer, allocation]() {
        if (buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(alloc, buffer, allocation);
        }
    });
    ctx.buffers.destroy(handle);
}

auto destroy(RenderContext &ctx, QueryPoolHandle handle, u64 retire_value) -> void {
    auto impl = ctx.query_pools.get(handle);
    if (!impl) {
        return;
    }

    // Extract only the VkQueryPool handle
    VkQueryPool query_pool = impl->pool;

    ctx.destroy_queue.enqueue(retire_value, [alloc = ctx.allocator, query_pool]() {
        VmaAllocatorInfo info{};
        vmaGetAllocatorInfo(alloc, &info);
        vkDestroyQueryPool(info.device, query_pool, nullptr);
    });
    ctx.query_pools.destroy(handle);
}

auto destroy(RenderContext &ctx, PipelineHandle handle, u64 retire_value) -> void {
    auto impl = ctx.pipeline_pool.get(handle);
    if (!impl) {
        return;
    }

    // Extract pipeline and layout handles
    VkPipeline pipeline = impl->pipeline;
    VkPipelineLayout layout = impl->layout;

    ctx.destroy_queue.enqueue(retire_value, [context = &ctx, pipeline, layout]() {
        destruction::pipeline(context->get_device(), std::tuple{pipeline, layout});
    });
    ctx.pipeline_pool.destroy(handle);
}

auto destroy(RenderContext &ctx, ShaderHandle handle, u64 retire_value) -> void {
    auto impl = ctx.shaders.get(handle);
    if (!impl) {
        return;
    }

    VkShaderModule shader_module = *impl;

    ctx.destroy_queue.enqueue(retire_value, [alloc = ctx.allocator, shader_module]() {
        VmaAllocatorInfo info{};
        vmaGetAllocatorInfo(alloc, &info);
        vkDestroyShaderModule(info.device, shader_module, nullptr);
    });
    ctx.shaders.destroy(handle);
}

auto create(RenderContext &ctx, CompiledPipeline &&v) -> PipelineHandle { return ctx.create_pipeline(std::move(v)); }
auto create(RenderContext &ctx, Buffer &&v) -> BufferHandle { return ctx.create_buffer(std::move(v)); }
auto create(RenderContext &ctx, OffscreenTarget &&v) -> TextureHandle { return ctx.create_texture(std::move(v)); }
auto create(RenderContext &ctx, VkShaderModule &&v) -> ShaderHandle { return ctx.create_shader(std::move(v)); }
auto create(RenderContext &ctx, QueryPoolState &&v) -> QueryPoolHandle { return ctx.create_query_pool(std::move(v)); }
