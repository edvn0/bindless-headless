#pragma once

#include "Buffer.hxx"
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

struct RenderContext {
    VmaAllocator &allocator;
    DeferredDestroyQueue destroy_queue{};
    BindlessSet *bindless_set{nullptr};


    TexturePool textures{};
    auto create_texture(OffscreenTarget &&) -> TextureHandle;

    SamplerPool samplers{};
    auto create_sampler(VkSampler &&) -> SamplerHandle;
    auto create_sampler(VkSamplerCreateInfo, std::string_view) -> SamplerHandle;

    BufferPool buffers{};
    auto create_buffer(Buffer &&) -> BufferHandle;

    QueryPoolPool query_pools{};
    auto create_query_pool(QueryPoolState &&) -> QueryPoolHandle;

    auto device_address(BufferHandle) -> DeviceAddress;
    auto device_address(BufferHandle) const -> DeviceAddress;
    auto clear_all() -> void;

    [[nodiscard]] auto get_device() const -> VkDevice;
};

auto destroy(RenderContext &ctx, TextureHandle handle, u64 retire_value = UINT64_MAX) -> void;

auto destroy(RenderContext &ctx, SamplerHandle handle, u64 retire_value = UINT64_MAX) -> void;

auto destroy(RenderContext &ctx, BufferHandle handle, u64 retire_value = UINT64_MAX) -> void;

auto destroy(RenderContext &ctx, QueryPoolHandle handle, u64 retire_value = UINT64_MAX) -> void;

namespace util {


    template<typename Mesh>
    concept MeshLike = requires(Mesh m) {
        { m.vertex_buffer() };
        { m.index_buffer() };
    } || requires(Mesh m) {
        { m.vertex_buffer };
        { m.index_buffer };
    };

    struct MeshDrawHandles {
        Buffer *vertex_buffer;
        Buffer *index_buffer;
    };

    auto get_mesh_buffers(RenderContext &ctx, MeshLike auto const &mesh) -> MeshDrawHandles {
        BufferHandle vb_handle;
        BufferHandle ib_handle;

        if constexpr (requires { mesh.vertex_buffer(); }) {
            vb_handle = mesh.vertex_buffer();
            ib_handle = mesh.index_buffer();
        } else {
            vb_handle = mesh.vertex_buffer;
            ib_handle = mesh.index_buffer;
        }

        auto &&[vb, ib] = ctx.buffers.get_multiple(vb_handle, ib_handle);

        return MeshDrawHandles{
                .vertex_buffer = vb,
                .index_buffer = ib,
        };
    }

} // namespace util
