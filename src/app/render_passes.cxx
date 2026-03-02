#include "app/render_passes.hxx"
#include "AlignedRingBuffer.hxx"
#include "BindlessHeadless.hxx"
#include "Constants.hxx"
#include "CreateInfo.hxx"
#include "Pipelines.hxx"
#include "RenderContext.hxx"

namespace {
    constexpr auto begin_query_for_index = [](const RenderContext &c, const VkCommandBuffer cmd,
                                              const GraphicsIndex index,
                                              const QueryPoolHandle stats_pool) -> VkQueryPool {
        u32 query_idx = static_cast<u32>(index);
        const auto *qs = c.query_pools.get(stats_pool);
        vkCmdBeginQuery(cmd, qs->pool, query_idx, 0);
        return qs->pool;
    };
    constexpr auto end_query_for_index = [](const VkCommandBuffer cmd, const GraphicsIndex index,
                                            const VkQueryPool pool) -> void {
        u32 query_idx = static_cast<u32>(index);
        vkCmdEndQuery(cmd, pool, query_idx);
    };
    template<typename... Vs>
    auto fill_zeros(VkCommandBuffer cmd, RenderContext &ctx, u32 frame_index, AlignedRingBuffer<Vs> &...buffers) {
        (buffers.fill_zeros(cmd, ctx, frame_index), ...);
    }
} // namespace

namespace RP {
    namespace {

        FrameMarkers g_frame{};

        inline auto resolve(RenderContext &rc, QueryPoolHandle ts_h, QueryPoolHandle stats_h) -> Markers {
            auto &&[ts, stats] = rc.query_pools.get_multiple(ts_h, stats_h);
            return Markers{.ts = ts, .stats = stats};
        }

    } // namespace

    auto setup_render_passes_for_frame(AppContext &ctx, BoundedFrameIndex frame_index) -> void {
        g_frame.compute = resolve(ctx.gpu.ctx, ctx.pipes.compute_stats_pool[frame_index],
                                  ctx.pipes.compute_query_pool[frame_index]);
        g_frame.graphics = resolve(ctx.gpu.ctx, ctx.pipes.graphics_stats_pool[frame_index],
                                   ctx.pipes.graphics_query_pool[frame_index]);
    }

    auto get_frame_markers() -> FrameMarkers const & { return g_frame; }

    auto graphics_specification(GraphicsIndex idx) -> Specification {
        switch (idx) {
            case GraphicsIndex::PreDepth:
                return {u32(GraphicsStamp::PreDepthBegin), u32(GraphicsStamp::PreDepthEnd), u32(idx)};
            case GraphicsIndex::GBuffer:
                return {u32(GraphicsStamp::GbufferBegin), u32(GraphicsStamp::GbufferEnd), u32(idx)};
            case GraphicsIndex::Deferred:
                return {u32(GraphicsStamp::DeferredBegin), u32(GraphicsStamp::DeferredEnd), u32(idx)};
            case GraphicsIndex::Skybox:
                return {u32(GraphicsStamp::SkyboxBegin), u32(GraphicsStamp::SkyboxEnd), u32(idx)};
            case GraphicsIndex::Tonemap:
                return {u32(GraphicsStamp::TonemapBegin), u32(GraphicsStamp::TonemapEnd), u32(idx)};
            case GraphicsIndex::Present:
                return {u32(GraphicsStamp::PresentBegin), u32(GraphicsStamp::PresentEnd), u32(idx)};
            case GraphicsIndex::ShadowMap:
                return {u32(GraphicsStamp::DirectionalShadowMapBegin), u32(GraphicsStamp::DirectionalShadowMapEnd),
                        u32(idx)};
            default:
                break;
        }
        return {};
    }

    auto compute_specification(ComputeIndex idx) -> Specification {
        switch (idx) {
            case ComputeIndex::RotateGeometry:
                return {u32(ComputeStamp::RotateGeometryBegin), u32(ComputeStamp::RotateGeometryEnd), u32(idx)};
            case ComputeIndex::RotateLights:
                return {u32(ComputeStamp::RotateLightsBegin), u32(ComputeStamp::RotateLightsEnd), u32(idx)};
            case ComputeIndex::LightClustering:
                return {u32(ComputeStamp::LightClusteringBegin), u32(ComputeStamp::LightClusteringEnd), u32(idx)};
            case ComputeIndex::Ssao:
                return {u32(ComputeStamp::SsaoBegin), u32(ComputeStamp::SsaoEnd), u32(idx)};
            case ComputeIndex::SsaoBlur:
                return {u32(ComputeStamp::SsaoBlurBegin), u32(ComputeStamp::SsaoBlurEnd), u32(idx)};
            default:
                break;
        }
        return {};
    }

    Scope::Scope(VkCommandBuffer cmd, Markers m, Specification s, VkPipelineStageFlags2 ts_begin,
                 VkPipelineStageFlags2 ts_end) : cmd_{cmd}, m_{m}, s_{s}, ts_begin_{ts_begin}, ts_end_{ts_end} {

        if (m_.ts)
            vkCmdWriteTimestamp2(cmd_, ts_begin_, m_.ts->pool, s_.timestamp_begin);
        if (m_.stats)
            vkCmdBeginQuery(cmd_, m_.stats->pool, s_.stats_index, 0);
    }

    Scope::Scope(Scope &&o) noexcept :
        cmd_{o.cmd_}, m_{o.m_}, s_{o.s_}, ts_begin_{o.ts_begin_}, ts_end_{o.ts_end_}, active_{o.active_} {
        o.active_ = false;
    }

    Scope::~Scope() {
        if (!active_)
            return;
        if (m_.stats)
            vkCmdEndQuery(cmd_, m_.stats->pool, s_.stats_index);
        if (m_.ts)
            vkCmdWriteTimestamp2(cmd_, ts_end_, m_.ts->pool, s_.timestamp_end);
    }

} // namespace RP

auto run_rotation_pass(AppContext &ctx, const u32 bounded_frame_index, const u32 last_frame_index,
                       const DeviceAddress &point_lights_base_addr, const SubmitSynchronisation &sync) -> u64 {
    auto &&[gpu, pipes, res, ui] = ctx;

    const RotatePushConstant geo_pc{
            .delta_time = static_cast<float>(ui.dt),
            .rads_per_second = glm::radians(20.0f),
            .total_time = static_cast<f32>(ui.total_time),
            .count = res.instance_count(),
            .previous_frame_transforms = res.transforms_ring.slot_device_address(last_frame_index),
            .transforms = res.transforms_ring.slot_device_address(bounded_frame_index),
            .previous_point_lights = res.point_lights_ring.slot_device_address(last_frame_index),
            .point_lights = res.point_lights_ring.slot_device_address(bounded_frame_index),
            .static_point_lights = point_lights_base_addr,
    };

    const u64 geo_signal = submit_stage(
            gpu.tl_compute, gpu.device,
            [&](VkCommandBuffer cmd) {
                TRACY_GPU_ZONE(gpu.tracy_compute.ctx, cmd, "RotateGeometryGPU");
                auto &&[ts, stats_pool] = gpu.ctx.query_pools.get_multiple(
                        pipes.compute_query_pool[bounded_frame_index], pipes.compute_stats_pool[bounded_frame_index]);

                auto _ = RP::begin_compute(cmd, ComputeIndex::RotateGeometry);

                auto *pipe = gpu.ctx.pipeline_pool.get(pipes.cube_rotation_pipeline);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline);
                vkCmdPushConstants(cmd, pipe->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(geo_pc), &geo_pc);

                const u32 groups = (res.instance_count() + 63u) / 64u;
                vkCmdDispatch(cmd, groups, 1, 1);

                auto *cube_buffer = gpu.ctx.buffers.get(res.transforms_ring.handle());
                VkBufferMemoryBarrier2 barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                barrier.buffer = cube_buffer->buffer();
                barrier.offset = static_cast<VkDeviceSize>(res.transforms_ring.slot_offset_bytes(bounded_frame_index));
                barrier.size = static_cast<VkDeviceSize>(res.instance_count() * sizeof(glm::mat4x3));

                VkDependencyInfo dep{};
                dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.bufferMemoryBarrierCount = 1;
                dep.pBufferMemoryBarriers = &barrier;
                vkCmdPipelineBarrier2(cmd, &dep);
            },
            sync);

    const TimelineWait geo_wait{
            .value = geo_signal,
            .semaphore = gpu.tl_compute.timeline,
            .stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    };

    const RotatePushConstant light_pc{
            .delta_time = static_cast<float>(ui.dt),
            .rads_per_second = glm::radians(20.0f),
            .total_time = static_cast<f32>(ui.total_time),
            .count = static_cast<u32>(res.all_point_lights.size()),
            .previous_frame_transforms = res.transforms_ring.slot_device_address(last_frame_index),
            .transforms = res.transforms_ring.slot_device_address(bounded_frame_index),
            .previous_point_lights = res.point_lights_ring.slot_device_address(last_frame_index),
            .point_lights = res.point_lights_ring.slot_device_address(bounded_frame_index),
            .static_point_lights = point_lights_base_addr,
    };

    return submit_stage(
            gpu.tl_compute, gpu.device,
            [&](VkCommandBuffer cmd) {
                TRACY_GPU_ZONE(gpu.tracy_compute.ctx, cmd, "RotateLightsGPU");
                auto &&[ts, stats_pool] = gpu.ctx.query_pools.get_multiple(
                        pipes.compute_query_pool[bounded_frame_index], pipes.compute_stats_pool[bounded_frame_index]);

                auto _ = RP::begin_compute(cmd, ComputeIndex::RotateLights);

                auto *pipe = gpu.ctx.pipeline_pool.get(pipes.light_rotation_pipeline);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline);
                vkCmdPushConstants(cmd, pipe->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(light_pc), &light_pc);

                const u32 groups = (static_cast<u32>(res.all_point_lights.size()) + 63u) / 64u;
                vkCmdDispatch(cmd, groups, 1, 1);


                VkBufferMemoryBarrier2 barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                barrier.buffer = gpu.ctx.buffers.get(res.point_lights_ring.handle())->buffer();
                barrier.offset = 0;
                barrier.size = VK_WHOLE_SIZE;

                VkDependencyInfo dep{};
                dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.bufferMemoryBarrierCount = 1;
                dep.pBufferMemoryBarriers = &barrier;
                vkCmdPipelineBarrier2(cmd, &dep);
            },
            SubmitSynchronisation{.timeline_waits = std::span(&geo_wait, 1)});
}


auto run_predepth_pass(AppContext &ctx, VkExtent2D frame_extent,
                       std::span<const MeshInstanceRange> mesh_instance_ranges, std::span<const DrawRanges> ranges,
                       const u32 bounded_frame_index, const SubmitSynchronisation &sync) -> u64 {
    auto &&[gpu, pipes, res, ui] = ctx;

    return submit_stage(
            gpu.tl_graphics, gpu.device,
            [&](VkCommandBuffer cmd) {
                TRACY_GPU_ZONE(gpu.tracy_graphics.ctx, cmd, "Predepth");

                auto &&[ts, stats_pool] = gpu.ctx.query_pools.get_multiple(
                        pipes.graphics_query_pool[bounded_frame_index], pipes.graphics_stats_pool[bounded_frame_index]);

                auto &&[predepth, alpha] =
                        gpu.ctx.pipeline_pool.get_multiple(pipes.predepth_pipeline, pipes.predepth_alpha_pipeline);

                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, GraphicsStamp::PreDepthBegin);
                begin_stats(cmd, *stats_pool, GraphicsIndex::PreDepth);

                auto *depth = gpu.ctx.textures.get(res.depth);

                depth->transition_if_not_initialised(
                        cmd, VK_IMAGE_LAYOUT_GENERAL,
                        {VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT});

                VkRenderingAttachmentInfo depth_attachment{};
                depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                depth_attachment.imageView = depth->attachment_view;
                depth_attachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                depth_attachment.clearValue = {.depthStencil = {0.0f, 0}};

                VkRenderingInfo rendering_info{};
                rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                rendering_info.renderArea = {.offset = {0, 0}, .extent = {frame_extent.width, frame_extent.height}};
                rendering_info.layerCount = 1;
                rendering_info.pDepthAttachment = &depth_attachment;

                vkCmdBeginRendering(cmd, &rendering_info);

                auto &&[vp, sc] = viewport_scissors(frame_extent);
                vkCmdSetViewport(cmd, 0, 1, &vp);
                vkCmdSetScissor(cmd, 0, 1, &sc);
                vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_GREATER_OR_EQUAL);
                vkCmdSetDepthBounds(cmd, 0.0f, 1.0f);
                vkCmdSetCullMode(cmd, VK_CULL_MODE_BACK_BIT);
                vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);


                auto *indirect = gpu.ctx.buffers.get(res.indirect_ring.handle());

                for (auto i = 0U; i < mesh_instance_ranges.size(); ++i) {
                    const auto &mir = mesh_instance_ranges[i];
                    const auto &range = ranges[i];
                    const auto &mesh = res.meshes[mir.mesh_index];

                    auto &&[verts, idx, materials] =
                            gpu.ctx.buffers.get_multiple(mesh.pos_uv_buffer, mesh.index_buffer, mesh.material_buffer);

                    vkCmdBindIndexBuffer(cmd, idx->buffer(), 0, VK_INDEX_TYPE_UINT32);
                    const std::array<VkBuffer, 1> vert_bufs = {verts->buffer()};
                    const std::array<VkDeviceSize, 1> vert_offs = {0};
                    const VkDeviceSize vert_size = verts->size();
                    vkCmdBindVertexBuffers2(cmd, 0, 1, vert_bufs.data(), vert_offs.data(), &vert_size, nullptr);

                    PredepthPushConstants pc{
                            .ubo = res.frame_ubo_ring.slot_device_address(bounded_frame_index),
                            .transforms = res.transforms_ring.slot_device_address(bounded_frame_index),
                            .draw_material_ids = res.draw_material_id_ring.slot_device_address(bounded_frame_index),
                            .materials = materials->device_address(),
                            .base_draw_id = 0,
                            .sampler_index = 0,
                    };

                    if (range.opaque_count > 0) {
                        pc.base_draw_id = range.opaque_base;
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, predepth->pipeline);
                        vkCmdPushConstants(cmd, predepth->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, predepth->layout, 0, 1,
                                                &gpu.bindless.set, 0, nullptr);
                        const VkDeviceSize offset =
                                static_cast<VkDeviceSize>(res.indirect_ring.slot_offset_bytes(bounded_frame_index)) +
                                range.opaque_base * sizeof(VkDrawIndexedIndirectCommand);

                        vkCmdDrawIndexedIndirect(cmd, indirect->buffer(), offset, range.opaque_count,
                                                 sizeof(VkDrawIndexedIndirectCommand));
                    }

                    if (range.alpha_count > 0) {
                        pc.base_draw_id = range.alpha_base;
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, alpha->pipeline);
                        vkCmdPushConstants(cmd, alpha->layout,
                                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc),
                                           &pc);
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, alpha->layout, 0, 1,
                                                &gpu.bindless.set, 0, nullptr);
                        const VkDeviceSize offset =
                                static_cast<VkDeviceSize>(res.indirect_ring.slot_offset_bytes(bounded_frame_index)) +
                                range.alpha_base * sizeof(VkDrawIndexedIndirectCommand);

                        vkCmdDrawIndexedIndirect(cmd, indirect->buffer(), offset, range.alpha_count,
                                                 sizeof(VkDrawIndexedIndirectCommand));
                    }
                }

                vkCmdEndRendering(cmd);
                end_stats(cmd, *stats_pool, GraphicsIndex::PreDepth);
                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, GraphicsStamp::PreDepthEnd);
            },
            sync);
}

auto run_environment_skybox_pass(AppContext &ctx, VkExtent2D frame_extent, BoundedFrameIndex bounded_frame_index,
                                 const SubmitSynchronisation &sync) -> TimelineValue {
    auto &&[gpu, pipes, res, ui] = ctx;

    return submit_stage(
            gpu.tl_graphics, gpu.device,
            [&](VkCommandBuffer cmd) {
                TRACY_GPU_ZONE(gpu.tracy_graphics.ctx, cmd, "EnvironmentSkybox");

                auto &&[ts, stats_pool] = gpu.ctx.query_pools.get_multiple(
                        pipes.graphics_query_pool[bounded_frame_index], pipes.graphics_stats_pool[bounded_frame_index]);

                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, GraphicsStamp::SkyboxBegin);
                begin_stats(cmd, *stats_pool, GraphicsIndex::Skybox);

                auto *skybox_pipe = gpu.ctx.pipeline_pool.get(pipes.skybox_pipeline);
                auto *lit = gpu.ctx.textures.get(res.lit_hdr);
                auto *depth = gpu.ctx.textures.get(res.depth);

                const std::array<VkImageMemoryBarrier2, 2> barriers{{
                        VkImageMemoryBarrier2{
                                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .pNext = nullptr,
                                .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                .srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = lit->image,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                        },
                        VkImageMemoryBarrier2{
                                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .pNext = nullptr,
                                .srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                                VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                .srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                                .dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = depth->image,
                                .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
                        },
                }};

                VkDependencyInfo dep{};
                dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.imageMemoryBarrierCount = static_cast<u32>(barriers.size());
                dep.pImageMemoryBarriers = barriers.data();
                vkCmdPipelineBarrier2(cmd, &dep);

                VkRenderingAttachmentInfo lit_att{};
                lit_att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                lit_att.imageView = lit->attachment_view;
                lit_att.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                lit_att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                lit_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

                VkRenderingAttachmentInfo depth_att{};
                depth_att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                depth_att.imageView = depth->attachment_view;
                depth_att.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                depth_att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                depth_att.storeOp = VK_ATTACHMENT_STORE_OP_NONE;

                VkRenderingInfo ri{};
                ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                ri.renderArea = {.offset = {0, 0}, .extent = frame_extent};
                ri.layerCount = 1;
                ri.colorAttachmentCount = 1;
                ri.pColorAttachments = &lit_att;
                ri.pDepthAttachment = &depth_att;

                vkCmdBeginRendering(cmd, &ri);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skybox_pipe->pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skybox_pipe->layout, 0, 1,
                                        &gpu.bindless.set, 0, nullptr);

                auto &&[vp, sc] = viewport_scissors(frame_extent);
                vkCmdSetViewport(cmd, 0, 1, &vp);
                vkCmdSetScissor(cmd, 0, 1, &sc);
                vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_GREATER_OR_EQUAL);
                vkCmdSetDepthBounds(cmd, 0.0f, 1.0f);
                vkCmdSetCullMode(cmd, VK_CULL_MODE_NONE);
                vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);

                SkyboxPushConstants pc{
                        .frame_ubo = res.frame_ubo_ring.slot_device_address(bounded_frame_index),
                        .cubemap_index = res.environment_cubemap.index(),
                        .sampler_index = pipes.linear_clamp.index(),
                };

                vkCmdPushConstants(cmd, skybox_pipe->layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0, sizeof(pc), &pc);

                vkCmdDraw(cmd, 3, 1, 0, 0);

                vkCmdEndRendering(cmd);

                end_stats(cmd, *stats_pool, GraphicsIndex::Skybox);
                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, GraphicsStamp::SkyboxEnd);
            },
            sync);
}

auto run_light_clustering_pass(AppContext &ctx, const u32 bounded_frame_index, const SubmitSynchronisation &sync)
        -> u64 {
    auto &&[gpu, pipes, res, ui] = ctx;


    return submit_stage(
            gpu.tl_compute, gpu.device,
            [&](VkCommandBuffer cmd) {
                {
                    auto barrier = create_info<VkMemoryBarrier2>();
                    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
                    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;

                    auto dep_info = create_info<VkDependencyInfo>();
                    dep_info.memoryBarrierCount = 1;
                    dep_info.pMemoryBarriers = &barrier;
                    vkCmdPipelineBarrier2(cmd, &dep_info);
                }

                TRACY_GPU_ZONE(gpu.tracy_compute.ctx, cmd, "ClusteredLightCulling");

                auto &&[cqs, css] = gpu.ctx.query_pools.get_multiple(pipes.compute_query_pool[bounded_frame_index],
                                                                     pipes.compute_stats_pool[bounded_frame_index]);

                write_ts(cmd, *cqs, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, ComputeStamp::LightClusteringBegin);
                begin_stats(cmd, *css, ComputeIndex::LightClustering);

                const ClusteredLightCullingPushConstants pc{
                        .frame_ubo = res.frame_ubo_ring.slot_device_address(bounded_frame_index),
                        .all_lights = res.point_lights_ring.slot_device_address(bounded_frame_index),
                        .mesh_indirect = res.mesh_indirect_ring.slot_device_address(bounded_frame_index),
                        .clusters = res.clusters.slot_device_address(bounded_frame_index),
                        .cluster_light_indices = res.cluster_light_indices.slot_device_address(bounded_frame_index),

                        .z_near = res.clustering_config.z_near,
                        .z_far = res.clustering_config.z_far,
                        .log_z_scale = res.clustering_config.log_z_scale,

                        .light_count = res.light_count,
                        .tiles_x = res.clustering_config.tiles_x,
                        .tiles_y = res.clustering_config.tiles_y,
                        .tiles_z = res.clustering_config.tiles_z,
                        .cluster_count = res.clustering_config.cluster_count,

                };

                auto &&[build_pipe, finalise] = gpu.ctx.pipeline_pool.get_multiple(pipes.cluster_build_groups_pipeline,
                                                                                   pipes.finalise_compact_pipeline);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, build_pipe->pipeline);
                vkCmdPushConstants(cmd, build_pipe->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cmd, res.clustering_config.cluster_count, 1, 1);
                VkMemoryBarrier2 mem_barrier{};
                mem_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                mem_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                mem_barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
                mem_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                mem_barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;

                VkDependencyInfo dep_info{};
                dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep_info.memoryBarrierCount = 1;
                dep_info.pMemoryBarriers = &mem_barrier;
                vkCmdPipelineBarrier2(cmd, &dep_info);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, finalise->pipeline);
                vkCmdPushConstants(cmd, finalise->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cmd, 1, 1, 1);
                vkCmdPipelineBarrier2(cmd, &dep_info);

                // --------------- DEBUG ------------------
                auto *heatmap_pipe = gpu.ctx.pipeline_pool.get(pipes.debug_light_clustering);
                const u32 cell_size = 16;
                const u32 slices_per_row = 4; // arrange Z slices into a 4x4 grid

                const u32 hm_width = res.clustering_config.tiles_x * slices_per_row * cell_size; // 16*4*16 = 1024
                const u32 hm_height = res.clustering_config.tiles_y * (res.clustering_config.tiles_z / slices_per_row) *
                                      cell_size; // 9*4*16 = 576

                const HeatmapPushConstants hm_pc{
                        .clusters = res.clusters.slot_device_address(bounded_frame_index),
                        .tiles_x = res.clustering_config.tiles_x,
                        .tiles_y = res.clustering_config.tiles_y,
                        .tiles_z = res.clustering_config.tiles_z,
                        .max_lights_per_cluster = max_lights_per_cluster,
                        .debug_texture_uav_index = res.debug_culling.index(),
                        .cell_size = cell_size,
                        .slices_per_row = slices_per_row,
                };

                auto *debug_tex = gpu.ctx.textures.get(res.debug_culling);
                debug_tex->transition_if_not_initialised(
                        cmd, VK_IMAGE_LAYOUT_GENERAL,
                        {VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT});

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, heatmap_pipe->pipeline);
                vkCmdPushConstants(cmd, heatmap_pipe->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(hm_pc), &hm_pc);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, heatmap_pipe->layout, 0, 1,
                                        &gpu.bindless.set, 0, nullptr);
                const u32 gx = (hm_width + 7) / 8;
                const u32 gy = (hm_height + 7) / 8;
                vkCmdDispatch(cmd, gx, gy, 1);
                // ----------------------------------------


                end_stats(cmd, *css, ComputeIndex::LightClustering);
                write_ts(cmd, *cqs, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, ComputeStamp::LightClusteringEnd);
            },
            sync);
}

auto run_directional_shadow_map_pass(AppContext &ctx, std::span<const MeshInstanceRange> mesh_instance_ranges,
                                     std::span<const DrawRanges> ranges, const u32 bounded_frame_index,
                                     const SubmitSynchronisation &sync) -> u64 {
    auto &&[gpu, pipes, res, ui] = ctx;

    return submit_stage(
            gpu.tl_graphics, gpu.device,
            [&](VkCommandBuffer cmd) {
                TRACY_GPU_ZONE(gpu.tracy_graphics.ctx, cmd, "DirectionalShadowMap");

                auto &&[ts, stats_pool] = gpu.ctx.query_pools.get_multiple(
                        pipes.graphics_query_pool[bounded_frame_index], pipes.graphics_stats_pool[bounded_frame_index]);

                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, GraphicsStamp::DirectionalShadowMapBegin);
                begin_stats(cmd, *stats_pool, GraphicsIndex::ShadowMap);

                auto *shadow_pipeline = gpu.ctx.pipeline_pool.get(pipes.directional_shadow_map_pipeline);
                auto *shadow_depth = gpu.ctx.textures.get(res.directional_shadow_map_depth);
                auto shadow_extent = shadow_depth->extent();

                shadow_depth->transition_if_not_initialised(
                        cmd, VK_IMAGE_LAYOUT_GENERAL,
                        {VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT});

                VkRenderingAttachmentInfo depth_attachment{};
                depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                depth_attachment.imageView = shadow_depth->attachment_view;
                depth_attachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                depth_attachment.clearValue = {.depthStencil = {0.0f, 0}};

                VkRenderingInfo rendering_info{};
                rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                rendering_info.renderArea = {.offset = {0, 0}, .extent = shadow_extent};
                rendering_info.layerCount = 1;
                rendering_info.pDepthAttachment = &depth_attachment;

                vkCmdBeginRendering(cmd, &rendering_info);

                auto &&[vp, sc] = viewport_scissors(shadow_extent);
                vkCmdSetViewport(cmd, 0, 1, &vp);
                vkCmdSetScissor(cmd, 0, 1, &sc);
                vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_GREATER_OR_EQUAL);
                vkCmdSetDepthBounds(cmd, 0.0f, 1.0f);
                vkCmdSetCullMode(cmd, VK_CULL_MODE_BACK_BIT);
                vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);
                vkCmdSetDepthBiasEnable(cmd, VK_TRUE);
                vkCmdSetDepthBias(cmd, ui.shadow_config.depth_bias_constant_factor, ui.shadow_config.depth_bias_clamp,
                                  ui.shadow_config.depth_bias_slope_factor);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline->pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline->layout, 0, 1,
                                        &gpu.bindless.set, 0, nullptr);

                auto *indirect = gpu.ctx.buffers.get(res.indirect_ring.handle());

                for (usize i = 0; i < mesh_instance_ranges.size(); ++i) {
                    const auto &mir = mesh_instance_ranges[i];
                    const auto &range = ranges[i];
                    const auto &mesh = res.meshes[mir.mesh_index];

                    auto &&[verts, idx, materials] =
                            gpu.ctx.buffers.get_multiple(mesh.pos_uv_buffer, mesh.index_buffer, mesh.material_buffer);

                    vkCmdBindIndexBuffer(cmd, idx->buffer(), 0, VK_INDEX_TYPE_UINT32);
                    const std::array<VkBuffer, 1> vert_bufs = {verts->buffer()};
                    constexpr std::array<VkDeviceSize, 1> vert_offs = {0};
                    const VkDeviceSize vert_size = verts->size();
                    vkCmdBindVertexBuffers2(cmd, 0, 1, vert_bufs.data(), vert_offs.data(), &vert_size, nullptr);

                    ShadowMapPushConstants pc{
                            .light_view_proj = ui.shadow_config.light_view_proj,
                            .transforms = res.transforms_ring.slot_device_address(bounded_frame_index),
                            .draw_material_ids = res.draw_material_id_ring.slot_device_address(bounded_frame_index),
                            .materials = materials->device_address(),
                            .base_draw_id = 0,
                            .sampler_index = pipes.depth_compare_filter.index(),
                    };

                    if (range.opaque_count > 0) {
                        pc.base_draw_id = range.opaque_base;
                        vkCmdPushConstants(cmd, shadow_pipeline->layout,
                                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc),
                                           &pc);

                        const VkDeviceSize offset =
                                static_cast<VkDeviceSize>(res.indirect_ring.slot_offset_bytes(bounded_frame_index)) +
                                range.opaque_base * sizeof(VkDrawIndexedIndirectCommand);

                        vkCmdDrawIndexedIndirect(cmd, indirect->buffer(), offset, range.opaque_count,
                                                 sizeof(VkDrawIndexedIndirectCommand));
                    }
                }

                vkCmdEndRendering(cmd);

                VkImageMemoryBarrier2 shadow_to_read{};
                shadow_to_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                shadow_to_read.srcStageMask =
                        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
                shadow_to_read.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                shadow_to_read.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                shadow_to_read.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                shadow_to_read.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                shadow_to_read.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                shadow_to_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                shadow_to_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                shadow_to_read.image = shadow_depth->image;
                shadow_to_read.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

                VkDependencyInfo dep{};
                dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.imageMemoryBarrierCount = 1;
                dep.pImageMemoryBarriers = &shadow_to_read;
                vkCmdPipelineBarrier2(cmd, &dep);

                end_stats(cmd, *stats_pool, GraphicsIndex::ShadowMap);
                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, GraphicsStamp::DirectionalShadowMapEnd);
            },
            sync);
}

auto run_gbuffer_pass(AppContext &ctx, VkExtent2D frame_extent, std::span<const MeshInstanceRange> mesh_instance_ranges,
                      std::span<const DrawRanges> ranges, const u32 bounded_frame_index,
                      const SubmitSynchronisation &sync) -> u64 {
    auto &&[gpu, pipes, res, ui] = ctx;

    return submit_stage(
            gpu.tl_graphics, gpu.device,
            [&](VkCommandBuffer cmd) {
                TRACY_GPU_ZONE(gpu.tracy_graphics.ctx, cmd, "GBuffer MRT");

                auto *ts = gpu.ctx.query_pools.get(pipes.graphics_query_pool[bounded_frame_index]);
                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, GraphicsStamp::GbufferBegin);
                auto *pool = begin_query_for_index(ctx.gpu.ctx, cmd, GraphicsIndex::GBuffer,
                                                   pipes.graphics_stats_pool[bounded_frame_index]);

                auto &&[mrt_pipeline] = gpu.ctx.pipeline_pool.get_multiple(pipes.gbuffer_pipeline_mrt);

                auto *g0 = gpu.ctx.textures.get(res.gbuffer0);
                auto *g1 = gpu.ctx.textures.get(res.gbuffer1);
                auto *g2 = gpu.ctx.textures.get(res.gbuffer2);
                auto *depth = gpu.ctx.textures.get(res.depth);

                g0->transition_if_not_initialised(
                        cmd, VK_IMAGE_LAYOUT_GENERAL,
                        {VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT});
                g1->transition_if_not_initialised(
                        cmd, VK_IMAGE_LAYOUT_GENERAL,
                        {VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT});
                g2->transition_if_not_initialised(
                        cmd, VK_IMAGE_LAYOUT_GENERAL,
                        {VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT});
                depth->transition_if_not_initialised(
                        cmd, VK_IMAGE_LAYOUT_GENERAL,
                        {VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT});

                VkRenderingAttachmentInfo colors[3]{};
                auto init_color = [&](VkRenderingAttachmentInfo &a, VkImageView view) {
                    a.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    a.imageView = view;
                    a.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    a.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                    a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                    a.clearValue = {.color = {.float32 = {0, 0, 0, 0}}};
                };
                init_color(colors[0], g0->attachment_view);
                init_color(colors[1], g1->attachment_view);
                init_color(colors[2], g2->attachment_view);

                VkRenderingAttachmentInfo depth_att{};
                depth_att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                depth_att.imageView = depth->attachment_view;
                depth_att.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                depth_att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                depth_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                depth_att.clearValue = {.depthStencil = {0.0f, 0}};

                VkRenderingInfo ri{};
                ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                ri.renderArea = {.offset = {0, 0}, .extent = frame_extent};
                ri.layerCount = 1;
                ri.colorAttachmentCount = 3;
                ri.pColorAttachments = colors;
                ri.pDepthAttachment = &depth_att;

                vkCmdBeginRendering(cmd, &ri);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mrt_pipeline->pipeline);

                auto &&[vp, sc] = viewport_scissors(frame_extent);
                vkCmdSetViewport(cmd, 0, 1, &vp);
                vkCmdSetScissor(cmd, 0, 1, &sc);
                vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_EQUAL);
                vkCmdSetCullMode(cmd, VK_CULL_MODE_BACK_BIT);
                vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);
                vkCmdSetDepthBounds(cmd, 0.0f, 1.0f);

                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mrt_pipeline->layout, 0, 1,
                                        &gpu.bindless.set, 0, nullptr);

                auto *indirect = gpu.ctx.buffers.get(res.indirect_ring.handle());

                for (usize i = 0; i < mesh_instance_ranges.size(); ++i) {
                    const auto &mir = mesh_instance_ranges[i];
                    const auto &range = ranges[i];
                    const auto &mesh = res.meshes[mir.mesh_index];

                    auto &&[verts, idx, materials] =
                            gpu.ctx.buffers.get_multiple(mesh.vertex_buffer, mesh.index_buffer, mesh.material_buffer);

                    vkCmdBindIndexBuffer(cmd, idx->buffer(), 0, VK_INDEX_TYPE_UINT32);
                    VkBuffer vb = verts->buffer();
                    VkDeviceSize off = 0;
                    vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &off);

                    RenderingPushConstants pc{
                            .ubo = res.frame_ubo_ring.slot_device_address(bounded_frame_index),
                            .transforms = res.transforms_ring.slot_device_address(bounded_frame_index),
                            .draw_material_ids = res.draw_material_id_ring.slot_device_address(bounded_frame_index),
                            .materials = materials->device_address(),
                            .base_draw_id = 0,
                            .sampler_index = pipes.linear_repeat.index(),
                    };

                    if (range.opaque_count > 0) {
                        pc.base_draw_id = range.opaque_base;
                        vkCmdPushConstants(cmd, mrt_pipeline->layout,
                                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc),
                                           &pc);

                        const VkDeviceSize offset =
                                static_cast<VkDeviceSize>(res.indirect_ring.slot_offset_bytes(bounded_frame_index)) +
                                static_cast<VkDeviceSize>(range.opaque_base) * sizeof(VkDrawIndexedIndirectCommand);

                        vkCmdDrawIndexedIndirect(cmd, indirect->buffer(), offset, range.opaque_count,
                                                 sizeof(VkDrawIndexedIndirectCommand));
                    }

                    if (range.alpha_count > 0) {
                        pc.base_draw_id = range.alpha_base;
                        vkCmdPushConstants(cmd, mrt_pipeline->layout,
                                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc),
                                           &pc);

                        const VkDeviceSize offset =
                                static_cast<VkDeviceSize>(res.indirect_ring.slot_offset_bytes(bounded_frame_index)) +
                                static_cast<VkDeviceSize>(range.alpha_base) * sizeof(VkDrawIndexedIndirectCommand);

                        vkCmdDrawIndexedIndirect(cmd, indirect->buffer(), offset, range.alpha_count,
                                                 sizeof(VkDrawIndexedIndirectCommand));
                    }
                }

                vkCmdEndRendering(cmd);
                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, GraphicsStamp::GbufferEnd);
                end_query_for_index(cmd, GraphicsIndex::GBuffer, pool);
            },
            sync);
}

auto run_ssao_pass(AppContext &ctx, VkExtent2D frame_extent, BoundedFrameIndex bounded_frame_index,
                   const SubmitSynchronisation &sync) -> TimelineValue {
    auto &&[gpu, pipes, res, ui] = ctx;

    return submit_stage(
            gpu.tl_compute, gpu.device,
            [&](VkCommandBuffer cmd) {
                TRACY_GPU_ZONE(gpu.tracy_compute.ctx, cmd, "SSAO");

                auto &&[ts, stats_pool] = gpu.ctx.query_pools.get_multiple(
                        pipes.compute_query_pool[bounded_frame_index], pipes.compute_stats_pool[bounded_frame_index]);

                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, ComputeStamp::SsaoBegin);
                begin_stats(cmd, *stats_pool, ComputeIndex::Ssao);

                auto *ssao_tex = gpu.ctx.textures.get(res.ssao_output);
                auto *g0 = gpu.ctx.textures.get(res.gbuffer0);
                auto *g1 = gpu.ctx.textures.get(res.gbuffer1);
                auto *depth_tex = gpu.ctx.textures.get(res.depth);

                ssao_tex->transition_if_not_initialised(
                        cmd, VK_IMAGE_LAYOUT_GENERAL,
                        {VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT});

                const std::array acquire_barriers{
                        VkImageMemoryBarrier2{
                                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .pNext = nullptr,
                                .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .srcAccessMask = VK_ACCESS_2_NONE,
                                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = g0->image,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                        },
                        VkImageMemoryBarrier2{
                                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .pNext = nullptr,
                                .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .srcAccessMask = VK_ACCESS_2_NONE,
                                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = g1->image,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                        },
                        VkImageMemoryBarrier2{
                                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .pNext = nullptr,
                                .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .srcAccessMask = VK_ACCESS_2_NONE,
                                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = depth_tex->image,
                                .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
                        },
                };

                VkDependencyInfo dep_acquire{};
                dep_acquire.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep_acquire.imageMemoryBarrierCount = static_cast<u32>(acquire_barriers.size());
                dep_acquire.pImageMemoryBarriers = acquire_barriers.data();
                vkCmdPipelineBarrier2(cmd, &dep_acquire);

                auto *pipe = gpu.ctx.pipeline_pool.get(pipes.ssao_pipeline);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->layout, 0, 1, &gpu.bindless.set, 0,
                                        nullptr);

                const auto hemisphere_address = gpu.ctx.buffers.get(res.ssao_hemisphere_kernel)->device_address();
                const auto noise_address = gpu.ctx.buffers.get(res.noise_ssao_kernel)->device_address();

                const SSAOPushConstants pc{
                        .frame_ubo = res.frame_ubo_ring.slot_device_address(bounded_frame_index),
                        .hemisphere_kernel = hemisphere_address,
                        .noise_kernel = noise_address,
                        .gbuffer0_index = res.gbuffer0.index(),
                        .gbuffer1_index = res.gbuffer1.index(),
                        .depth_index = res.depth.index(),
                        .ssao_output_index = res.ssao_output.index(),
                        .sampler_index = pipes.linear_clamp.index(),
                        .radius = 0.5f,
                        .bias = 0.025f,
                };

                vkCmdPushConstants(cmd, pipe->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

                const u32 group_x = (frame_extent.width + 7u) / 8u;
                const u32 group_y = (frame_extent.height + 7u) / 8u;
                vkCmdDispatch(cmd, group_x, group_y, 1);

                // Release-side barrier: make the SSAO write visible before the timeline signal,
                // so deferred lighting's acquire on the graphics queue sees it.
                const VkImageMemoryBarrier2 release_barrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                        .pNext = nullptr,
                        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        .dstAccessMask = VK_ACCESS_2_NONE,
                        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = ssao_tex->image,
                        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                };

                VkDependencyInfo dep_release{};
                dep_release.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep_release.imageMemoryBarrierCount = 1;
                dep_release.pImageMemoryBarriers = &release_barrier;
                vkCmdPipelineBarrier2(cmd, &dep_release);

                end_stats(cmd, *stats_pool, ComputeIndex::Ssao);
                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, ComputeStamp::SsaoEnd);
            },
            sync);
}

auto run_ssao_blur_pass(AppContext &ctx, VkExtent2D frame_extent, BoundedFrameIndex bounded_frame_index,
                        const SubmitSynchronisation &sync) -> TimelineValue {
    auto &&[gpu, pipes, res, ui] = ctx;

    return submit_stage(
            gpu.tl_compute, gpu.device,
            [&](VkCommandBuffer cmd) {
                TRACY_GPU_ZONE(gpu.tracy_compute.ctx, cmd, "SSAO Blur");

                auto &&[ts, stats_pool] = gpu.ctx.query_pools.get_multiple(
                        pipes.compute_query_pool[bounded_frame_index], pipes.compute_stats_pool[bounded_frame_index]);

                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, ComputeStamp::SsaoBlurBegin);
                begin_stats(cmd, *stats_pool, ComputeIndex::SsaoBlur);

                auto *ssao_in = gpu.ctx.textures.get(res.ssao_output);
                auto *blur_tmp = gpu.ctx.textures.get(res.ssao_blurred_temp);
                auto *ssao_out = gpu.ctx.textures.get(res.ssao_blurred);
                auto *depth_tex = gpu.ctx.textures.get(res.depth);

                blur_tmp->transition_if_not_initialised(
                        cmd, VK_IMAGE_LAYOUT_GENERAL,
                        {VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT});
                ssao_out->transition_if_not_initialised(
                        cmd, VK_IMAGE_LAYOUT_GENERAL,
                        {VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT});

                const std::array acquire_barriers{
                        VkImageMemoryBarrier2{
                                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .pNext = nullptr,
                                .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = ssao_in->image,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                        },
                        VkImageMemoryBarrier2{
                                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .pNext = nullptr,
                                .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .srcAccessMask = VK_ACCESS_2_NONE,
                                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = depth_tex->image,
                                .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
                        },
                };

                VkDependencyInfo dep_acquire{};
                dep_acquire.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep_acquire.imageMemoryBarrierCount = static_cast<u32>(acquire_barriers.size());
                dep_acquire.pImageMemoryBarriers = acquire_barriers.data();
                vkCmdPipelineBarrier2(cmd, &dep_acquire);

                auto *pipe = gpu.ctx.pipeline_pool.get(pipes.ssao_blur_pipeline);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->layout, 0, 1, &gpu.bindless.set, 0,
                                        nullptr);

                const auto group_x = (frame_extent.width + 7u) / 8u;
                const auto group_y = (frame_extent.height + 7u) / 8u;

                // --- Horizontal pass: ssao_output -> ssao_blurred_temp ---
                const SSAOBlurPushConstants pc_h{
                        .ssao_input_index = res.ssao_output.index(),
                        .ssao_output_index = res.ssao_blurred_temp.index(),
                        .depth_index = res.depth.index(),
                        .sampler_index = pipes.linear_clamp.index(),
                        .horizontal = 1,
                };
                vkCmdPushConstants(cmd, pipe->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_h), &pc_h);
                vkCmdDispatch(cmd, group_x, group_y, 1);

                // Barrier: tmp written by horizontal, read by vertical
                const VkImageMemoryBarrier2 mid_barrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                        .pNext = nullptr,
                        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = blur_tmp->image,
                        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                };
                VkDependencyInfo dep_mid{};
                dep_mid.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep_mid.imageMemoryBarrierCount = 1;
                dep_mid.pImageMemoryBarriers = &mid_barrier;
                vkCmdPipelineBarrier2(cmd, &dep_mid);

                // --- Vertical pass: ssao_blurred_temp -> ssao_blurred ---
                const SSAOBlurPushConstants pc_v{
                        .ssao_input_index = res.ssao_blurred_temp.index(),
                        .ssao_output_index = res.ssao_blurred.index(),
                        .depth_index = res.depth.index(),
                        .sampler_index = pipes.linear_clamp.index(),
                        .horizontal = 0,
                };
                vkCmdPushConstants(cmd, pipe->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_v), &pc_v);
                vkCmdDispatch(cmd, group_x, group_y, 1);

                // Release ssao_blurred to graphics queue for deferred lighting
                const VkImageMemoryBarrier2 release_barrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                        .pNext = nullptr,
                        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        .dstAccessMask = VK_ACCESS_2_NONE,
                        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .srcQueueFamilyIndex = gpu.queue_family_indices.compute,
                        .dstQueueFamilyIndex = gpu.queue_family_indices.graphics,
                        .image = ssao_out->image,
                        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                };
                VkDependencyInfo dep_release{};
                dep_release.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep_release.imageMemoryBarrierCount = 1;
                dep_release.pImageMemoryBarriers = &release_barrier;
                vkCmdPipelineBarrier2(cmd, &dep_release);

                end_stats(cmd, *stats_pool, ComputeIndex::SsaoBlur);
                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, ComputeStamp::SsaoBlurEnd);
            },
            sync);
}

auto run_deferred_lighting_pass(AppContext &ctx, const VkExtent2D frame_extent, const u32,
                                const u32 bounded_frame_index, const SubmitSynchronisation &sync) -> u64 {
    auto &&[gpu, pipes, res, ui] = ctx;

    return submit_stage(
            gpu.tl_graphics, gpu.device,
            [&](VkCommandBuffer cmd) {
                TRACY_GPU_ZONE(gpu.tracy_graphics.ctx, cmd, "DeferredLighting(FS)");

                auto &&ts = gpu.ctx.query_pools.get(pipes.graphics_query_pool[bounded_frame_index]);
                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, GraphicsStamp::DeferredBegin);
                auto *pool = begin_query_for_index(ctx.gpu.ctx, cmd, GraphicsIndex::Deferred,
                                                   pipes.graphics_stats_pool[bounded_frame_index]);

                auto &&[mrt_lighting, debug_point_light] = gpu.ctx.pipeline_pool.get_multiple(
                        pipes.gbuffer_pipeline_lighting, pipes.debug_point_light_pipeline);

                // auto *indirect_buffer = gpu.ctx.buffers.get(res.mesh_indirect_ring.handle());

                auto *g0 = gpu.ctx.textures.get(res.gbuffer0);
                auto *g1 = gpu.ctx.textures.get(res.gbuffer1);
                auto *g2 = gpu.ctx.textures.get(res.gbuffer2);
                auto *depth = gpu.ctx.textures.get(res.depth);
                auto *lit = gpu.ctx.textures.get(res.lit_hdr);

                g0->transition_if_not_initialised(
                        cmd, VK_IMAGE_LAYOUT_GENERAL,
                        {VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT});
                g1->transition_if_not_initialised(
                        cmd, VK_IMAGE_LAYOUT_GENERAL,
                        {VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT});
                g2->transition_if_not_initialised(
                        cmd, VK_IMAGE_LAYOUT_GENERAL,
                        {VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT});
                depth->transition_if_not_initialised(
                        cmd, VK_IMAGE_LAYOUT_GENERAL,
                        {VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT});
                lit->transition_if_not_initialised(
                        cmd, VK_IMAGE_LAYOUT_GENERAL,
                        {VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT});

                const std::array<VkImageMemoryBarrier2, 5> barriers{
                        VkImageMemoryBarrier2{
                                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .pNext = nullptr,
                                .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = g0->image,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                        },
                        VkImageMemoryBarrier2{
                                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .pNext = nullptr,
                                .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = g1->image,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                        },
                        VkImageMemoryBarrier2{
                                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .pNext = nullptr,
                                .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = g2->image,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                        },
                        VkImageMemoryBarrier2{
                                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                // depth was written in predepth/gbuffer depth test
                                .pNext = nullptr,
                                .srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                                VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                .srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = depth->image,
                                .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
                        },
                        VkImageMemoryBarrier2{
                                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .pNext = nullptr,
                                .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                                .srcAccessMask = VK_ACCESS_2_NONE,
                                .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = lit->image,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                        },
                };

                VkDependencyInfo dep{};
                dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.imageMemoryBarrierCount = static_cast<u32>(barriers.size());
                dep.pImageMemoryBarriers = barriers.data();
                vkCmdPipelineBarrier2(cmd, &dep);

                VkRenderingAttachmentInfo lit_att{};
                lit_att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                lit_att.imageView = lit->attachment_view;
                lit_att.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                lit_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                lit_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                lit_att.clearValue = {.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}};

                VkRenderingInfo ri{};
                ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                ri.renderArea = {.offset = {0, 0}, .extent = frame_extent};
                ri.layerCount = 1;
                ri.colorAttachmentCount = 1;
                ri.pColorAttachments = &lit_att;

                vkCmdBeginRendering(cmd, &ri);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mrt_lighting->pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mrt_lighting->layout, 0, 1,
                                        &gpu.bindless.set, 0, nullptr);

                auto &&[vp, sc] = viewport_scissors(frame_extent);
                vkCmdSetViewport(cmd, 0, 1, &vp);
                vkCmdSetScissor(cmd, 0, 1, &sc);

                DeferredLightingPushConstants pc{
                        .frame_ubo = res.frame_ubo_ring.slot_device_address(bounded_frame_index),
                        .point_lights = res.point_lights_ring.slot_device_address(bounded_frame_index),
                        .clusters = res.clusters.slot_device_address(bounded_frame_index),
                        .cluster_light_indices = res.cluster_light_indices.slot_device_address(bounded_frame_index),
                        .shadow_matrix = ui.shadow_config.light_view_proj,
                        .log_z_scale = res.clustering_config.log_z_scale,
                        .near_plane = z_near,

                        .tiles_x = res.clustering_config.tiles_x,
                        .tiles_y = res.clustering_config.tiles_y,
                        .tiles_z = res.clustering_config.tiles_z,

                        .gbuffer0_index = res.gbuffer0.index(),
                        .gbuffer1_index = res.gbuffer1.index(),
                        .gbuffer2_index = res.gbuffer2.index(),
                        .ssao_index = res.ssao_blurred.index(),
                        .depth_index = res.depth.index(),
                        .sampler_index = pipes.linear_clamp.index(),
                        .shadow_texture_index = res.directional_shadow_map_depth.index(),
                        .shadow_sampler_index = pipes.depth_compare_filter.index(),
                        .debug_mode = static_cast<u32>(ui.debug_mode),
                };

                vkCmdPushConstants(cmd, mrt_lighting->layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0, sizeof(pc), &pc);

                vkCmdDraw(cmd, 3, 1, 0, 0);

                vkCmdEndRendering(cmd);

                VkImageMemoryBarrier2 lit_to_read{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                        .pNext = nullptr,
                        .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = lit->image,
                        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                };

                VkDependencyInfo dep2{};
                dep2.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep2.imageMemoryBarrierCount = 1;
                dep2.pImageMemoryBarriers = &lit_to_read;
                vkCmdPipelineBarrier2(cmd, &dep2);

                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, GraphicsStamp::DeferredEnd);
                end_query_for_index(cmd, GraphicsIndex::Deferred, pool);
            },
            sync);
}

auto run_tonemap_pass(AppContext &ctx, const VkExtent2D frame_extent, const u32 bounded_frame_index,
                      const SubmitSynchronisation &sync) -> u64 {
    auto &&[gpu, pipes, res, ui] = ctx;

    return submit_stage(
            gpu.tl_graphics, gpu.device,
            [&](VkCommandBuffer cmd) {
                TRACY_GPU_ZONE(gpu.tracy_graphics.ctx, cmd, "Tonemapping");

                auto *ts = gpu.ctx.query_pools.get(pipes.graphics_query_pool[bounded_frame_index]);
                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, GraphicsStamp::TonemapBegin);

                auto *pool = begin_query_for_index(ctx.gpu.ctx, cmd, GraphicsIndex::Tonemap,
                                                   pipes.graphics_stats_pool[bounded_frame_index]);

                auto *tonemap = gpu.ctx.pipeline_pool.get(pipes.tonemap_pipeline);

                auto *hdr = gpu.ctx.textures.get(res.lit_hdr);
                auto *ldr = gpu.ctx.textures.get(res.tonemapped);

                // GENERAL everywhere: make sure resources are at least initialised to GENERAL (UNDEFINED -> GENERAL
                // once)
                hdr->transition_if_not_initialised(cmd, VK_IMAGE_LAYOUT_GENERAL,
                                                   {VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                                                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT});

                ldr->transition_if_not_initialised(
                        cmd, VK_IMAGE_LAYOUT_GENERAL,
                        {VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT});

                // Order: previous HDR writes -> fragment sampling in tonemap
                // (layout stays GENERAL; this is purely an execution+access barrier)
                auto hdr_to_sample = create_info<VkImageMemoryBarrier2>();
                hdr_to_sample.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                hdr_to_sample.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                hdr_to_sample.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                hdr_to_sample.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                hdr_to_sample.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                hdr_to_sample.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                hdr_to_sample.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                hdr_to_sample.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                hdr_to_sample.image = hdr->image;
                hdr_to_sample.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

                auto dep_hdr = create_info<VkDependencyInfo>();
                dep_hdr.imageMemoryBarrierCount = 1;
                dep_hdr.pImageMemoryBarriers = &hdr_to_sample;
                vkCmdPipelineBarrier2(cmd, &dep_hdr);

                // Dynamic rendering with GENERAL layout
                auto color_attachment = create_info<VkRenderingAttachmentInfo>();
                color_attachment.imageView =
                        ldr->attachment_view; // must match pipeline rendering format (R8G8B8A8_UNORM)
                color_attachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                color_attachment.clearValue = {.color = {.float32 = {0.0f, 0.0f, 0.0f, 0.0f}}};

                auto ri = create_info<VkRenderingInfo>();
                ri.renderArea = {.offset = {0, 0}, .extent = frame_extent};
                ri.layerCount = 1;
                ri.colorAttachmentCount = 1;
                ri.pColorAttachments = &color_attachment;

                vkCmdBeginRendering(cmd, &ri);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tonemap->pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tonemap->layout, 0, 1, &gpu.bindless.set,
                                        0, nullptr);

                TonemapPushConstants pc{
                        .exposure = 1.0f,
                        .image_index = res.lit_hdr.index(),
                        .sampler_index = pipes.linear_clamp.index(),
                };

                auto &&[vp, sc] = viewport_scissors(frame_extent);
                vkCmdSetViewport(cmd, 0, 1, &vp);
                vkCmdSetScissor(cmd, 0, 1, &sc);

                vkCmdPushConstants(cmd, tonemap->layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                   sizeof(pc), &pc);

                vkCmdDraw(cmd, 3, 1, 0, 0);

                vkCmdEndRendering(cmd);

                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, GraphicsStamp::TonemapEnd);
                end_query_for_index(cmd, GraphicsIndex::Tonemap, pool);
            },
            sync);
}

auto run_swapchain_pass(AppContext &ctx, const u32 swap_image_index, const u32 bounded_frame_index,
                        const SubmitSynchronisation &sync) -> u64 {
    auto &&[gpu, pipes, res, ui] = ctx;

    return submit_stage(
            gpu.tl_graphics, gpu.device,
            [&](VkCommandBuffer cmd) {
                TRACY_GPU_ZONE(gpu.tracy_graphics.ctx, cmd, "Present+ImGui");

                auto *ts = gpu.ctx.query_pools.get(pipes.graphics_query_pool[bounded_frame_index]);
                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, GraphicsStamp::PresentBegin);

                auto *pool = begin_query_for_index(ctx.gpu.ctx, cmd, GraphicsIndex::Present,
                                                   pipes.graphics_stats_pool[bounded_frame_index]);

                auto *tonemapped = gpu.ctx.textures.get(res.tonemapped);

                VkImage dst_image = gpu.swapchain.image(swap_image_index);
                VkImageView dst_view = gpu.swapchain.image_view(swap_image_index); // identity swizzle

                tonemapped->transition_if_not_initialised(cmd, VK_IMAGE_LAYOUT_GENERAL,
                                                          {VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                                                           VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT});

                auto tonemap_to_sample = create_info<VkImageMemoryBarrier2>();
                tonemap_to_sample.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                tonemap_to_sample.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                tonemap_to_sample.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                tonemap_to_sample.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                tonemap_to_sample.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                tonemap_to_sample.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                tonemap_to_sample.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                tonemap_to_sample.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                tonemap_to_sample.image = tonemapped->image;
                tonemap_to_sample.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                      .baseMipLevel = 0,
                                                      .levelCount = 1,
                                                      .baseArrayLayer = 0,
                                                      .layerCount = 1};

                auto dep_tm = create_info<VkDependencyInfo>();
                dep_tm.imageMemoryBarrierCount = 1;
                dep_tm.pImageMemoryBarriers = &tonemap_to_sample;
                vkCmdPipelineBarrier2(cmd, &dep_tm);

                // Swapchain image: transition to GENERAL for rendering
                auto to_general = create_info<VkImageMemoryBarrier2>();
                to_general.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
                to_general.srcAccessMask = VK_ACCESS_2_NONE;
                to_general.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                to_general.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                to_general.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; // ok if you don't track; otherwise PRESENT_SRC_KHR
                to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                to_general.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                to_general.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                to_general.image = dst_image;
                to_general.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                               .baseMipLevel = 0,
                                               .levelCount = 1,
                                               .baseArrayLayer = 0,
                                               .layerCount = 1};

                auto dep_to_general = create_info<VkDependencyInfo>();
                dep_to_general.imageMemoryBarrierCount = 1;
                dep_to_general.pImageMemoryBarriers = &to_general;
                vkCmdPipelineBarrier2(cmd, &dep_to_general);

                auto color_attachment = create_info<VkRenderingAttachmentInfo>();
                color_attachment.imageView = dst_view;
                color_attachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                color_attachment.clearValue = {.color = {.float32 = {0.0f, 0.0f, 0.0f, 0.0f}}};

                auto ri = create_info<VkRenderingInfo>();
                ri.renderArea = {.offset = {0, 0}, .extent = gpu.swapchain.extent()};
                ri.layerCount = 1;
                ri.colorAttachmentCount = 1;
                ri.pColorAttachments = &color_attachment;

                vkCmdBeginRendering(cmd, &ri);

                ui.gui->render(cmd);

                vkCmdEndRendering(cmd);

                auto to_present = create_info<VkImageMemoryBarrier2>();
                to_present.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                to_present.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                to_present.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
                to_present.dstAccessMask = VK_ACCESS_2_NONE;
                to_present.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
                to_present.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                to_present.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                to_present.image = dst_image;
                to_present.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                               .baseMipLevel = 0,
                                               .levelCount = 1,
                                               .baseArrayLayer = 0,
                                               .layerCount = 1};

                auto dep_to_present = create_info<VkDependencyInfo>();
                dep_to_present.imageMemoryBarrierCount = 1;
                dep_to_present.pImageMemoryBarriers = &to_present;
                vkCmdPipelineBarrier2(cmd, &dep_to_present);

                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, GraphicsStamp::PresentEnd);
                end_query_for_index(cmd, GraphicsIndex::Present, pool);
            },
            sync);
}
