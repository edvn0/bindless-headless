#pragma once

#include "Buffer.hxx"
#include "PipelineCache.hxx"
#include "Pool.hxx"
#include "Types.hxx"


namespace EngineShaderIndices {
    inline constexpr u32 fullscreen_vertex_shader = 0;
}

struct QueueInfo {
    VkQueue queue;
    u32 family_index;
};

struct AllQueues {
    QueueInfo graphics;
    QueueInfo compute;
    QueueInfo transfer;
};

struct RenderContext {
    VmaAllocator allocator;
    DeferredDestroyQueue destroy_queue{};
    BindlessSet *bindless_set{nullptr};
    std::unique_ptr<GlobalCommandContext> command_ctx;
    std::unique_ptr<PipelineCache> pipeline_cache;
    AllQueues queues{};

    TexturePool textures{};
    auto create_texture(OffscreenTarget &&) -> TextureHandle;

    SamplerPool samplers{};
    auto create_sampler(VkSampler &&) -> SamplerHandle;
    auto create_sampler(VkSamplerCreateInfo, std::string_view) -> SamplerHandle;

    SamplerPool comparison_samplers{};
    auto create_comparison_sampler(VkSampler &&) -> SamplerHandle;
    auto create_comparison_sampler(VkSamplerCreateInfo, std::string_view) -> SamplerHandle;

    BufferPool buffers{};
    auto create_buffer(Buffer &&) -> BufferHandle;

    QueryPoolPool query_pools{};
    auto create_query_pool(QueryPoolState &&) -> QueryPoolHandle;

    PipelinePool pipeline_pool{};
    auto create_pipeline(CompiledPipeline &&) -> PipelineHandle;

    ShaderPool shaders{};
    auto create_shader(VkShaderModule &&) -> ShaderHandle;

    auto device_address(BufferHandle) -> DeviceAddress;
    auto flush_mapped_memory(BufferHandle, std::size_t offset, std::size_t size) const -> void;
    [[nodiscard]] auto texture_format(TextureHandle) const -> VkFormat;
    [[nodiscard]] auto device_address(BufferHandle) const -> DeviceAddress;
    [[nodiscard]] auto get_mapped_pointer(BufferHandle) const -> tl::expected<void *, Error>;

    auto clear_all() -> void;

    [[nodiscard]] auto get_device() const -> VkDevice;
    [[nodiscard]] auto get_physical_device() const -> VkPhysicalDevice;
    [[nodiscard]] auto get_instance() const -> VkInstance;
};

auto destroy(RenderContext &ctx, TextureHandle handle, u64 retire_value = UINT64_MAX) -> void;
auto destroy(RenderContext &ctx, SamplerHandle handle, u64 retire_value = UINT64_MAX) -> void;
auto destroy(RenderContext &ctx, BufferHandle handle, u64 retire_value = UINT64_MAX) -> void;
auto destroy(RenderContext &ctx, QueryPoolHandle handle, u64 retire_value = UINT64_MAX) -> void;
auto destroy(RenderContext &ctx, PipelineHandle handle, u64 retire_value = UINT64_MAX) -> void;
auto destroy(RenderContext &ctx, ShaderHandle handle, u64 retire_value = UINT64_MAX) -> void;
auto create(RenderContext &, CompiledPipeline &&) -> PipelineHandle;
auto create(RenderContext &, Buffer &&) -> BufferHandle;
auto create(RenderContext &, OffscreenTarget &&) -> TextureHandle;
auto create(RenderContext &, VkShaderModule &&) -> ShaderHandle;
auto create(RenderContext &, QueryPoolState &&) -> QueryPoolHandle;
