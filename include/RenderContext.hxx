#pragma once

#include "Buffer.hxx"
#include "PipelineCache.hxx"
#include "Pipelines.hxx"
#include "Pool.hxx"
#include "Types.hxx"


struct QueryPoolState {
    VkQueryPool pool = VK_NULL_HANDLE;
    u32 query_count = 0;
    double timestamp_period_ns = 1.0; // from VkPhysicalDeviceLimits::timestampPeriod
};

using TextureHandle = Handle<struct TextureTag>;
using TexturePool = Pool<TextureTag, OffscreenTarget>;
using SamplerHandle = Handle<struct SamplerTag>;
using SamplerPool = Pool<SamplerTag, VkSampler>;
using BufferHandle = Handle<struct BufferTag>;
using BufferPool = Pool<BufferTag, Buffer>;
using QueryPoolHandle = Handle<struct QueryPoolTag>;
using QueryPoolPool = Pool<QueryPoolTag, QueryPoolState>;
using PipelineHandle = Handle<struct PipelineTag>;
using PipelinePool = Pool<PipelineTag, CompiledPipeline>;

struct RenderContext {
    VmaAllocator &allocator;
    DeferredDestroyQueue destroy_queue{};
    BindlessSet *bindless_set{nullptr};
    std::unique_ptr<PipelineCache> pipeline_cache;

    TexturePool textures{};
    auto create_texture(OffscreenTarget &&) -> TextureHandle;

    SamplerPool samplers{};
    auto create_sampler(VkSampler &&) -> SamplerHandle;
    auto create_sampler(VkSamplerCreateInfo, std::string_view) -> SamplerHandle;

    BufferPool buffers{};
    auto create_buffer(Buffer &&) -> BufferHandle;

    QueryPoolPool query_pools{};
    auto create_query_pool(QueryPoolState &&) -> QueryPoolHandle;

    PipelinePool pipeline_pool{};
    auto create_pipeline(CompiledPipeline &&) -> PipelineHandle;

    auto device_address(BufferHandle) -> DeviceAddress;
    auto flush_mapped_memory(BufferHandle, std::size_t offset, std::size_t size) const -> void;
    [[nodiscard]] auto texture_format(TextureHandle) const -> VkFormat;
    [[nodiscard]] auto device_address(BufferHandle) const -> DeviceAddress;
    [[nodiscard]] auto get_mapped_pointer(BufferHandle) const -> tl::expected<void *, Error>;

    auto clear_all() -> void;

    [[nodiscard]] auto get_device() const -> VkDevice;
};

auto destroy(RenderContext &ctx, TextureHandle handle, u64 retire_value = UINT64_MAX) -> void;
auto destroy(RenderContext &ctx, SamplerHandle handle, u64 retire_value = UINT64_MAX) -> void;
auto destroy(RenderContext &ctx, BufferHandle handle, u64 retire_value = UINT64_MAX) -> void;
auto destroy(RenderContext &ctx, QueryPoolHandle handle, u64 retire_value = UINT64_MAX) -> void;
auto destroy(RenderContext &ctx, PipelineHandle handle, u64 retire_value = UINT64_MAX) -> void;
