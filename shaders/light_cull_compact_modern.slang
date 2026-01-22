import common;
import maths;

[[vk::constant_id(0)]]
const int MAX_WAVES_PER_GROUP = 4;
[[vk::constant_id(1)]]
const int THREADS_PER_GROUP = 64;

// ============================================================================
// Push constants (bound per-dispatch)
// ============================================================================
struct PushConstants
{
    UBO*        frame_ubo;
    PointLight* lights;

    uint*       flags;                // [light_count]
    uint*       prefix;               // [light_count] (inclusive, per-group)

    PointLight* compact;              // output compacted lights
    uint*       culled_light_count;   // single uint (global atomic counter)

    uint        light_count;
};

// ============================================================================
// Pass 1: compute flags[i] ∈ {0,1}
// ============================================================================
[numthreads(THREADS_PER_GROUP, 1, 1)]
[shader("compute")]
void LightFlagsCS(
    uint3 dtid : SV_DispatchThreadID,
    uniform PushConstants pc)
{
    uint idx = dtid.x;
    if (idx >= pc.light_count)
        return;

    float4x4 view = pc.frame_ubo->view;

    PointLight l = pc.lights[idx];
    pc.flags[idx] =
        light_in_frustum(l, view, pc.frame_ubo->frustum_planes) ? 1u : 0u;
}

// ============================================================================
// Pass 2: local scan + compaction with atomic reservation
// ============================================================================
groupshared uint g_wave_sums[MAX_WAVES_PER_GROUP];
groupshared uint g_group_base;   // global base offset for this group

[numthreads(THREADS_PER_GROUP, 1, 1)]
[shader("compute")]
void LightCompactCS(
    uint3 dtid : SV_DispatchThreadID,
    uint3 gid  : SV_GroupID,
    uint  gidx : SV_GroupIndex,
    uniform PushConstants pc)
{
    uint global_idx = dtid.x;

    // Load flag (0 if out of range)
    uint v = 0u;
    if (global_idx < pc.light_count)
        v = pc.flags[global_idx];

    // ------------------------------------------------------------------------
    // Wave-level prefix
    // ------------------------------------------------------------------------
    uint lane      = WaveGetLaneIndex();
    uint wave_size = WaveGetLaneCount();
    uint wave_id   = gidx / wave_size;

    uint wave_prefix = WavePrefixSum(v);

    // Last lane in each wave writes wave sum
    if (wave_id < MAX_WAVES_PER_GROUP && lane == wave_size - 1u)
        g_wave_sums[wave_id] = wave_prefix + v;

    GroupMemoryBarrierWithGroupSync();

    // ------------------------------------------------------------------------
    // Scan wave sums (single wave does this)
    // ------------------------------------------------------------------------
    uint wave_count = (THREADS_PER_GROUP + wave_size - 1u) / wave_size;

    if (gidx < wave_count)
    {
        uint acc = 0u;
        [unroll]
        for (uint k = 0u; k < gidx; ++k)
            acc += g_wave_sums[k];

        g_wave_sums[gidx] += acc;
    }

    GroupMemoryBarrierWithGroupSync();

    // ------------------------------------------------------------------------
    // Compute inclusive prefix within the group
    // ------------------------------------------------------------------------
    uint wave_offset = (wave_id > 0u) ? g_wave_sums[wave_id - 1u] : 0u;
    uint inclusive   = wave_prefix + wave_offset + v;

    if (global_idx < pc.light_count)
        pc.prefix[global_idx] = inclusive;

    // ------------------------------------------------------------------------
    // One atomic per group: reserve global space
    // ------------------------------------------------------------------------
    if (gidx == 0)
    {
        uint group_total = g_wave_sums[wave_count - 1u];
        InterlockedAdd(pc.culled_light_count[0], group_total, g_group_base);
    }

    GroupMemoryBarrierWithGroupSync();

    // ------------------------------------------------------------------------
    // Write compacted lights
    // ------------------------------------------------------------------------
    if (v == 0u || global_idx >= pc.light_count)
        return;

    uint dst = g_group_base + inclusive - 1u;

    PointLight l = pc.lights[global_idx];
    pc.compact[dst].position_radius  = l.position_radius;
    pc.compact[dst].colour_intensity = l.colour_intensity;
}
