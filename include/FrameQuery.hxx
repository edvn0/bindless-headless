#pragma once

#include "RenderContext.hxx"
#include "Types.hxx"

#include <volk.h>

template<typename Stamp>
static inline auto write_ts(VkCommandBuffer cmd, const QueryPoolState &qs, VkPipelineStageFlags2 stage, Stamp s)
        -> void {
    vkCmdWriteTimestamp2(cmd, stage, qs.pool, static_cast<u32>(s));
}

static inline auto begin_stats(VkCommandBuffer cmd, const QueryPoolState &qs, const auto query) -> void {
    vkCmdBeginQuery(cmd, qs.pool, static_cast<u32>(query), 0);
}

static inline auto end_stats(VkCommandBuffer cmd, const QueryPoolState &qs, const auto query) -> void {
    vkCmdEndQuery(cmd, qs.pool, static_cast<u32>(query));
}


static constexpr auto read_timestamp_pair_ms_any = [](const RenderContext &render_context, QueryPoolHandle h,
                                                      const auto begin_idx,
                                                      const auto end_idx) -> std::optional<double> {
    const auto *qs = render_context.query_pools.get(h);
    if (!qs) {
        return std::nullopt;
    }

    const u32 count = qs->query_count;
    if (static_cast<u32>(begin_idx) >= count || static_cast<u32>(end_idx) >= count) {
        return std::nullopt;
    }

    std::vector<u64> stamps(count, 0);

    const VkResult r =
            vkGetQueryPoolResults(render_context.get_device(), qs->pool, 0, count, stamps.size() * sizeof(u64),
                                  stamps.data(), sizeof(u64), VK_QUERY_RESULT_64_BIT);

    if (r == VK_NOT_READY) {
        return std::nullopt;
    }
    if (r != VK_SUCCESS) {
        return std::nullopt;
    }

    const u64 dt_ticks = stamps[static_cast<u32>(end_idx)] - stamps[static_cast<u32>(begin_idx)];
    const double dt_ns = static_cast<double>(dt_ticks) * qs->timestamp_period_ns;
    return dt_ns * 1e-6;
};

static constexpr auto read_timestamp_pairs_ms = [](const RenderContext &render_context,
                                                   QueryPoolHandle h) -> std::optional<std::vector<double>> {
    const auto *qs = render_context.query_pools.get(h);
    if (!qs) {
        return std::nullopt;
    }

    const u32 count = qs->query_count;
    if (count < 2 || (count % 2) != 0) {
        return std::nullopt;
    }

    std::vector<u64> stamps(count, 0);

    const VkResult r =
            vkGetQueryPoolResults(render_context.get_device(), qs->pool, 0, count, stamps.size() * sizeof(u64),
                                  stamps.data(), sizeof(u64), VK_QUERY_RESULT_64_BIT);

    if (r == VK_NOT_READY) {
        return std::nullopt;
    }
    if (r != VK_SUCCESS) {
        return std::nullopt;
    }

    std::vector<double> out{};
    out.reserve(count / 2);

    for (u32 i = 0; i < count; i += 2) {
        const u64 dt_ticks = stamps[i + 1] - stamps[i];
        const double dt_ns = static_cast<double>(dt_ticks) * qs->timestamp_period_ns;
        out.push_back(dt_ns * 1e-6);
    }

    return out;
};


struct GraphicsGpuStats {
    u64 input_assembly_vertices;
    u64 input_assembly_primitives;
    u64 vertex_shader_invocations;
    u64 clipping_invocations;
    u64 clipping_primitives;
    u64 fragment_shader_invocations;
    u64 mesh_shader_invocations;
    u64 task_shader_invocations;
};

struct ComputeGpuStats {
    u64 compute_shader_invocations;
};


static constexpr auto read_graphics_stats = [](const RenderContext &ctx,
                                               QueryPoolHandle h) -> std::optional<std::vector<GraphicsGpuStats>> {
    const auto *qs = ctx.query_pools.get(h);
    if (!qs || qs->query_count == 0)
        return std::nullopt;

    const u32 count = qs->query_count;
    std::vector<GraphicsGpuStats> results(count);

    VkResult r = vkGetQueryPoolResults(ctx.get_device(), qs->pool, 0, count, count * sizeof(GraphicsGpuStats),
                                       results.data(),
                                       sizeof(GraphicsGpuStats), // Stride is the size of one full struct
                                       VK_QUERY_RESULT_64_BIT);

    if (r != VK_SUCCESS)
        return std::nullopt;
    return results;
};

static constexpr auto read_compute_stats = [](const RenderContext &ctx,
                                              const auto h) -> std::optional<std::vector<ComputeGpuStats>> {
    const auto *qs = ctx.query_pools.get(h);
    if (!qs)
        return std::nullopt;

    const u32 count = qs->query_count;
    std::vector<ComputeGpuStats> results(count);

    VkResult r = vkGetQueryPoolResults(ctx.get_device(), qs->pool, 0, count, count * sizeof(ComputeGpuStats),
                                       results.data(), sizeof(ComputeGpuStats), VK_QUERY_RESULT_64_BIT);

    if (r != VK_SUCCESS)
        return std::nullopt;
    return results;
};
