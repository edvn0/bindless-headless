#include "app/render_passes.hxx"
#include "vulkan/vulkan_core.h"


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
    auto fill_zeros(VkCommandBuffer cmd, auto &buffers_ctx, auto &&...buffer_handles) {
        (vkCmdFillBuffer(cmd, buffers_ctx.get(buffer_handles)->buffer(), 0, VK_WHOLE_SIZE, 0), ...);
    }
} // namespace

auto run_rotation_pass(AppContext &ctx, const u32 bounded_frame_index, const u32 last_frame_index,
                       const DeviceAddress &point_lights_base_addr, const SubmitSynchronisation &sync) -> u64 {
    auto &&[gpu, pipes, res, ui] = ctx;
    return submit_stage(
            gpu.tl_compute, gpu.device,
            [&](VkCommandBuffer cmd) {
                TRACY_GPU_ZONE(gpu.tracy_compute.ctx, cmd, "RotateCubesGPU");

                auto &&[ts, stats_pool] = gpu.ctx.query_pools.get_multiple(
                        pipes.compute_query_pool[bounded_frame_index], pipes.compute_stats_pool[bounded_frame_index]);

                auto *pipe = gpu.ctx.pipeline_pool.get(pipes.cube_rotation_pipeline);

                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, ComputeStamp::RotateBegin);
                begin_stats(cmd, *stats_pool, ComputeIndex::Rotate);

                auto *cube_buffer = gpu.ctx.buffers.get(res.transforms_ring.handle());

                RotateCubesPushConstant pc{
                        .cube_count = res.instance_count,
                        .delta_time = static_cast<float>(ui.dt),
                        .rads_per_second = glm::radians(20.0f),
                        .total_time = static_cast<f32>(ui.total_time),
                        .light_count = static_cast<u32>(res.all_point_lights.size()),
                        .transforms = res.transforms_ring.slot_device_address(bounded_frame_index),
                        .previous_frame_transforms = res.transforms_ring.slot_device_address(last_frame_index),
                        .point_lights = res.point_lights_ring.slot_device_address(bounded_frame_index),
                        .previous_point_lights = res.point_lights_ring.slot_device_address(last_frame_index),
                        .static_point_light_base = point_lights_base_addr,
                };

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline);
                vkCmdPushConstants(cmd, pipe->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

                const u32 work_items = std::max(res.instance_count, pc.light_count);
                const u32 groups = (work_items + 63u) / 64u;
                vkCmdDispatch(cmd, groups, 1, 1);

                end_stats(cmd, *stats_pool, ComputeIndex::Rotate);
                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, ComputeStamp::RotateEnd);

                std::array<VkBufferMemoryBarrier2, 2> barriers{};
                barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                barriers[0].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
                barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                barriers[0].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                barriers[0].buffer = cube_buffer->buffer();
                barriers[0].offset =
                        static_cast<VkDeviceSize>(res.transforms_ring.slot_offset_bytes(bounded_frame_index));
                barriers[0].size = static_cast<VkDeviceSize>(res.instance_count * sizeof(glm::mat4x3));

                barriers[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                barriers[1].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
                barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                barriers[1].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                barriers[1].buffer = gpu.ctx.buffers.get(res.point_lights_ring.handle())->buffer();
                barriers[1].offset = 0;
                barriers[1].size = VK_WHOLE_SIZE;

                VkDependencyInfo dep_info{};
                dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep_info.bufferMemoryBarrierCount = static_cast<u32>(barriers.size());
                dep_info.pBufferMemoryBarriers = barriers.data();
                vkCmdPipelineBarrier2(cmd, &dep_info);
            },
            sync);
}


auto run_predepth_pass(AppContext &ctx, VkExtent2D frame_extent, const DrawRanges &ranges,
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

                auto &&depth = gpu.ctx.textures.get(res.depth);
                auto &&[indirect, verts, idx, materials] =
                        gpu.ctx.buffers.get_multiple(res.indirect_ring.handle(), res.mesh.pos_uv_buffer,
                                                     res.mesh.index_buffer, res.mesh.material_buffer);

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

                const PredepthPushConstants pc{
                        .ubo = res.frame_ubo_ring.slot_device_address(bounded_frame_index),
                        .transforms = res.transforms_ring.slot_device_address(bounded_frame_index),
                        .draw_material_ids = res.draw_material_id_ring.slot_device_address(bounded_frame_index),
                        .materials = materials->device_address(),
                        .base_draw_id = ranges.opaque_base,
                        .sampler_index = 0,
                };

                auto &&[vp, sc] = viewport_scissors(frame_extent);
                vkCmdSetViewport(cmd, 0, 1, &vp);
                vkCmdSetScissor(cmd, 0, 1, &sc);
                vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_GREATER_OR_EQUAL); // Reverse-Z
                vkCmdSetDepthBounds(cmd, 0.0F, 1.0F);
                vkCmdSetCullMode(cmd, VK_CULL_MODE_BACK_BIT);
                vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);

                vkCmdBindIndexBuffer(cmd, idx->buffer(), 0, VK_INDEX_TYPE_UINT32);
                std::array<VkBuffer, 1> buffers = {verts->buffer()};
                std::array<VkDeviceSize, 1> offsets = {0};
                const auto size = VkDeviceSize{verts->size()};
                vkCmdBindVertexBuffers2(cmd, 0, 1, buffers.data(), offsets.data(), &size, nullptr);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gpu.bindless.pipeline_layout, 0, 1,
                                        &gpu.bindless.set, 0, nullptr);

                if (ranges.opaque_count > 0) {
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, predepth->pipeline);

                    PredepthPushConstants opaque_pc = pc;
                    opaque_pc.base_draw_id = ranges.opaque_base;

                    vkCmdPushConstants(cmd, predepth->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(opaque_pc),
                                       &opaque_pc);

                    const VkDeviceSize opaque_offset =
                            static_cast<VkDeviceSize>(res.indirect_ring.slot_offset_bytes(bounded_frame_index)) +
                            (ranges.opaque_base * sizeof(VkDrawIndexedIndirectCommand));

                    vkCmdDrawIndexedIndirect(cmd, indirect->buffer(), opaque_offset, ranges.opaque_count,
                                             sizeof(VkDrawIndexedIndirectCommand));
                }

                if (ranges.alpha_count > 0) {
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, alpha->pipeline);

                    PredepthPushConstants alpha_pc = pc;
                    alpha_pc.base_draw_id = ranges.alpha_base;

                    vkCmdPushConstants(cmd, alpha->layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                       sizeof(alpha_pc), &alpha_pc);

                    const VkDeviceSize alpha_offset =
                            static_cast<VkDeviceSize>(res.indirect_ring.slot_offset_bytes(bounded_frame_index)) +
                            (ranges.alpha_base * sizeof(VkDrawIndexedIndirectCommand));

                    vkCmdDrawIndexedIndirect(cmd, indirect->buffer(), alpha_offset, ranges.alpha_count,
                                             sizeof(VkDrawIndexedIndirectCommand));
                }

                vkCmdEndRendering(cmd);
                end_stats(cmd, *stats_pool, GraphicsIndex::PreDepth);
                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, GraphicsStamp::PreDepthEnd);
            },
            sync);
}

auto run_light_frustum_cull_pass(AppContext &ctx, const u32 bounded_frame_index, DeviceAddresses<4> &&device_addresses,
                                 const SubmitSynchronisation &sync) -> u64 {
    auto &&[gpu, pipes, res, ui] = ctx;
    auto &&[flags_addr, prefix_addr, compact_addr, culled_light_count_addr] = device_addresses;
    return submit_stage(
            gpu.tl_compute, gpu.device,
            [&](VkCommandBuffer cmd) {
                TRACY_GPU_ZONE(gpu.tracy_compute.ctx, cmd, "LightCulling");

                auto &&[cqs, css] = gpu.ctx.query_pools.get_multiple(pipes.compute_query_pool[bounded_frame_index],
                                                                     pipes.compute_stats_pool[bounded_frame_index]);

                write_ts(cmd, *cqs, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, ComputeStamp::CullBegin);
                begin_stats(cmd, *css, ComputeIndex::Cull);

                const PointLightCullingPushConstants pc{
                        .ubo = res.frame_ubo_ring.slot_device_address(bounded_frame_index),
                        .lights = res.point_lights_ring.slot_device_address(bounded_frame_index),
                        .flags = flags_addr,
                        .prefix = prefix_addr,
                        .compact = compact_addr,
                        .culled_light_count = culled_light_count_addr,
                        .light_count = res.light_count,
                };

                auto bind_and_dispatch = [&](auto &pl, u32 groups_x) {
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.layout, 0, 1, &gpu.bindless.set, 0,
                                            nullptr);

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl.pipeline);

                    vkCmdPushConstants(cmd, pl.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       sizeof(PointLightCullingPushConstants), &pc);

                    vkCmdDispatch(cmd, groups_x, 1u, 1u);
                };

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

                fill_zeros(cmd, gpu.ctx.buffers, res.flags, res.prefix, res.compact_lights, res.culled_light_count);

                vkCmdPipelineBarrier2(cmd, &dep_info);

                const u32 gc = (res.light_count + THREADS_PER_GROUP - 1) / THREADS_PER_GROUP;

                auto &&[flags, compact] =
                        gpu.ctx.pipeline_pool.get_multiple(pipes.flags_pipeline, pipes.compact_pipeline);

                bind_and_dispatch(*flags, gc);
                vkCmdPipelineBarrier2(cmd, &dep_info);

                bind_and_dispatch(*compact, gc);

                end_stats(cmd, *css, ComputeIndex::Cull);
                write_ts(cmd, *cqs, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, ComputeStamp::CullEnd);
            },
            sync);
}

auto run_light_clustering_pass(AppContext &ctx, const u32 bounded_frame_index, DeviceAddresses<4> &&device_addresses,
                               const SubmitSynchronisation &sync) -> u64 {
    auto &&[gpu, pipes, res, ui] = ctx;
    auto &&[compact_addr, culled_light_count_addr, clusters_addr, cluster_light_indices_addr] = device_addresses;

    return submit_stage(
            gpu.tl_compute, gpu.device,
            [&](VkCommandBuffer cmd) {
                TRACY_GPU_ZONE(gpu.tracy_compute.ctx, cmd, "ClusteredLightCulling");

                auto &&[cqs, css] = gpu.ctx.query_pools.get_multiple(pipes.compute_query_pool[bounded_frame_index],
                                                                     pipes.compute_stats_pool[bounded_frame_index]);

                write_ts(cmd, *cqs, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, ComputeStamp::ClusteringBegin);
                begin_stats(cmd, *css, ComputeIndex::Clustering);

                const ClusteredLightCullingPushConstants pc{
                        .frame_ubo = res.frame_ubo_ring.slot_device_address(bounded_frame_index),
                        .culled_lights = compact_addr,
                        .culled_light_count = culled_light_count_addr,

                        .z_near = res.clustering_config.z_near,
                        .z_far = res.clustering_config.z_far,
                        .log_z_scale = res.clustering_config.log_z_scale,

                        .tiles_x = res.clustering_config.tiles_x,
                        .tiles_y = res.clustering_config.tiles_y,
                        .tiles_z = res.clustering_config.tiles_z,
                        .cluster_count = res.clustering_config.cluster_count,

                        .clusters = clusters_addr,
                        .cluster_light_indices = cluster_light_indices_addr,
                };

                auto build_pipe = gpu.ctx.pipeline_pool.get(pipes.cluster_build_groups_pipeline);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, build_pipe->pipeline);
                vkCmdPushConstants(cmd, build_pipe->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cmd, res.clustering_config.cluster_count, 1, 1);

                end_stats(cmd, *css, ComputeIndex::Clustering);
                write_ts(cmd, *cqs, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, ComputeStamp::ClusteringEnd);
                TracyVkCollect(gpu.tracy_compute.ctx, cmd);
            },
            sync);
}

auto run_gbuffer_pass(AppContext &ctx, const VkExtent2D frame_extent, const DrawRanges &ranges,
                      const u32 bounded_frame_index, const SubmitSynchronisation &sync) -> u64 {
    auto &&[gpu, pipes, res, ui] = ctx;
    return submit_stage(
            gpu.tl_graphics, gpu.device,
            [&](VkCommandBuffer cmd) {
                TRACY_GPU_ZONE(gpu.tracy_graphics.ctx, cmd, "GBuffer MRT");

                auto *ts = gpu.ctx.query_pools.get(pipes.graphics_query_pool[bounded_frame_index]);
                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, GraphicsStamp::GbufferBegin);
                auto *pool = begin_query_for_index(ctx.gpu.ctx, cmd, GraphicsIndex::GBuffer,
                                                   pipes.graphics_stats_pool[bounded_frame_index]);

                auto *mrt_pipeline = gpu.ctx.pipeline_pool.get(pipes.gbuffer_pipeline_mrt);

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
                depth_att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // keep predepth
                depth_att.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; // no need to store
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
                vkCmdSetDepthCompareOp(cmd,
                                       VK_COMPARE_OP_EQUAL); // matches your predepth = GEQUAL reverseZ + load depth
                vkCmdSetCullMode(cmd, VK_CULL_MODE_BACK_BIT);
                vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);
                vkCmdSetDepthBounds(cmd, 0.0f, 1.0f);

                auto &&[indirect, verts, idx, materials] =
                        gpu.ctx.buffers.get_multiple(res.indirect_ring.handle(), res.mesh.vertex_buffer,
                                                     res.mesh.index_buffer, res.mesh.material_buffer);

                RenderingPushConstants pc{
                        .ubo = res.frame_ubo_ring.slot_device_address(bounded_frame_index),
                        .transforms = res.transforms_ring.slot_device_address(bounded_frame_index),
                        .draw_material_ids = res.draw_material_id_ring.slot_device_address(bounded_frame_index),
                        .materials = materials->device_address(),
                        .base_draw_id = ranges.opaque_base,
                        .sampler_index = pipes.linear_repeat.index(),
                };

                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mrt_pipeline->layout, 0, 1,
                                        &gpu.bindless.set, 0, nullptr);

                vkCmdPushConstants(cmd, mrt_pipeline->layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0, sizeof(pc), &pc);

                vkCmdBindIndexBuffer(cmd, idx->buffer(), 0, VK_INDEX_TYPE_UINT32);
                VkBuffer vb = verts->buffer();
                VkDeviceSize off = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &off);

                if (ranges.opaque_count > 0) {
                    pc.base_draw_id = ranges.opaque_base;
                    vkCmdPushConstants(cmd, mrt_pipeline->layout,
                                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

                    VkDeviceSize indirect_offset_bytes =
                            static_cast<VkDeviceSize>(res.indirect_ring.slot_offset_bytes(bounded_frame_index)) +
                            static_cast<VkDeviceSize>(ranges.opaque_base) * sizeof(VkDrawIndexedIndirectCommand);

                    vkCmdDrawIndexedIndirect(cmd, indirect->buffer(), indirect_offset_bytes, ranges.opaque_count,
                                             sizeof(VkDrawIndexedIndirectCommand));
                }

                if (ranges.alpha_count > 0) {
                    pc.base_draw_id = ranges.alpha_base;
                    vkCmdPushConstants(cmd, mrt_pipeline->layout,
                                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

                    VkDeviceSize indirect_offset_bytes =
                            static_cast<VkDeviceSize>(res.indirect_ring.slot_offset_bytes(bounded_frame_index)) +
                            static_cast<VkDeviceSize>(ranges.alpha_base) * sizeof(VkDrawIndexedIndirectCommand);

                    vkCmdDrawIndexedIndirect(cmd, indirect->buffer(), indirect_offset_bytes, ranges.alpha_count,
                                             sizeof(VkDrawIndexedIndirectCommand));
                }

                vkCmdEndRendering(cmd);

                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, GraphicsStamp::GbufferEnd);
                end_query_for_index(cmd, GraphicsIndex::GBuffer, pool);
            },
            sync);
}

auto run_deferred_lighting_pass(AppContext &ctx, const VkExtent2D frame_extent, DeviceAddresses<2> &&device_addresses,
                                const u32 bounded_frame_index, const SubmitSynchronisation &sync) -> u64 {
    auto &&[gpu, pipes, res, ui] = ctx;
    auto &&[clusters_addr, cluster_light_indices_addr] = device_addresses;
    return submit_stage(
            gpu.tl_graphics, gpu.device,
            [&](VkCommandBuffer cmd) {
                TRACY_GPU_ZONE(gpu.tracy_graphics.ctx, cmd, "DeferredLighting(FS)");

                auto &&ts = gpu.ctx.query_pools.get(pipes.graphics_query_pool[bounded_frame_index]);
                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, GraphicsStamp::DeferredBegin);
                auto *pool = begin_query_for_index(ctx.gpu.ctx, cmd, GraphicsIndex::Deferred,
                                                   pipes.graphics_stats_pool[bounded_frame_index]);

                auto mrt_lighting = gpu.ctx.pipeline_pool.get(pipes.gbuffer_pipeline_lighting);

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

                        .tiles_x = res.clustering_config.tiles_x,
                        .tiles_y = res.clustering_config.tiles_y,
                        .tiles_z = res.clustering_config.tiles_z,
                        .log_z_scale = res.clustering_config.log_z_scale,

                        .clusters = clusters_addr,
                        .cluster_light_indices = cluster_light_indices_addr,

                        .gbuffer0_index = res.gbuffer0.index(),
                        .gbuffer1_index = res.gbuffer1.index(),
                        .gbuffer2_index = res.gbuffer2.index(),
                        .depth_index = res.depth.index(),
                        .lit_hdr_uav_index = 0,
                        .debug_output_index = res.debug_culling.index(),
                        .sampler_index = pipes.linear_clamp.index(),
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
                color_attachment.clearValue = {.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}};

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
                auto *present = gpu.ctx.pipeline_pool.get(pipes.present_pipeline);

                VkImage dst_image = gpu.swapchain.image(swap_image_index);
                VkImageView dst_view = gpu.swapchain.image_view(swap_image_index); // identity swizzle

                // Ensure tonemapped is initialised to GENERAL if it was recreated (UNDEFINED -> GENERAL once)
                tonemapped->transition_if_not_initialised(cmd, VK_IMAGE_LAYOUT_GENERAL,
                                                          {VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                                                           VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT});

                // Order: tonemap writes -> fragment sampling in present shader (layout stays GENERAL)
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
                tonemap_to_sample.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

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
                to_general.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

                auto dep_to_general = create_info<VkDependencyInfo>();
                dep_to_general.imageMemoryBarrierCount = 1;
                dep_to_general.pImageMemoryBarriers = &to_general;
                vkCmdPipelineBarrier2(cmd, &dep_to_general);

                auto color_attachment = create_info<VkRenderingAttachmentInfo>();
                color_attachment.imageView = dst_view;
                color_attachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                color_attachment.clearValue = {.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}};

                auto ri = create_info<VkRenderingInfo>();
                ri.renderArea = {.offset = {0, 0}, .extent = gpu.swapchain.extent()};
                ri.layerCount = 1;
                ri.colorAttachmentCount = 1;
                ri.pColorAttachments = &color_attachment;

                vkCmdBeginRendering(cmd, &ri);

                // 1) Present fullscreen pass (samples tonemapped)
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, present->pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, present->layout, 0, 1, &gpu.bindless.set,
                                        0, nullptr);

                auto &&[vp, sc] = viewport_scissors(gpu.swapchain.extent());
                vkCmdSetViewport(cmd, 0, 1, &vp);
                vkCmdSetScissor(cmd, 0, 1, &sc);

                const bool swap_is_srgb = gpu.swapchain.format() == VK_FORMAT_B8G8R8A8_SRGB ||
                                          gpu.swapchain.format() == VK_FORMAT_R8G8B8A8_SRGB;

                PresentPushConstants pc{
                        .image_index = res.tonemapped.index(),
                        .sampler_index = pipes.linear_clamp.index(),
                        .dst_is_srgb = swap_is_srgb ? 1u : 0u,
                };

                vkCmdPushConstants(cmd, present->layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                   sizeof(pc), &pc);

                vkCmdDraw(cmd, 3, 1, 0, 0);

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
                to_present.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

                auto dep_to_present = create_info<VkDependencyInfo>();
                dep_to_present.imageMemoryBarrierCount = 1;
                dep_to_present.pImageMemoryBarriers = &to_present;
                vkCmdPipelineBarrier2(cmd, &dep_to_present);

                write_ts(cmd, *ts, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, GraphicsStamp::PresentEnd);
                end_query_for_index(cmd, GraphicsIndex::Present, pool);

                TracyVkCollect(gpu.tracy_graphics.ctx, cmd);
            },
            sync);
}
