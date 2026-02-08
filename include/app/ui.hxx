#pragma once
#include "Pool.hxx"
#include "RenderContext.hxx"
#include "Types.hxx"
#include "ui/PerformanceGraph.hxx"

auto draw_ui(PerformanceGraph<8, 120>& gpu_frame_graph,
             const RenderContext& ctx,
             std::span<QueryPoolHandle, frames_in_flight> compute_query_pool,
             std::span<QueryPoolHandle, frames_in_flight> compute_stats_pool,
             std::span<QueryPoolHandle, frames_in_flight> graphics_query_pool,
             std::span<QueryPoolHandle, frames_in_flight> graphics_stats_pool,
             uint32_t frame_index) -> void;
