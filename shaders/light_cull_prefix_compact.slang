import common;
import maths;

static const uint THREADS_PER_GROUP       = 64;
static const uint MAX_WAVES_PER_GROUP     = 4;

struct PushConstants
{
    UBO*        frame_ubo;
    PointLight* lights;
    uint*       flags;
    uint*       prefix;
    uint*       group_sums;
    uint*       group_offsets;
    PointLight* compact;
    IndirectCommand* indirect;
    IndirectMeshCommand* indirect_mesh;
    uint        image_index;
    uint        light_count;
    uint        group_count;
};

[[vk::push_constant]]
PushConstants g_push;

// -----------------------------------------------------------------------------
// Pass 1: compute flags[i] ∈ {0,1}
// -----------------------------------------------------------------------------
[numthreads(THREADS_PER_GROUP, 1, 1)]
[shader("compute")]
void LightFlagsCS(uint3 dtid : SV_DispatchThreadID)
{
    uint idx = dtid.x;
    if (idx >= g_push.light_count)
        return;

    float4x4 view = g_push.frame_ubo->view;

    PointLight l = g_push.lights[idx];
    g_push.flags[idx] = light_in_frustum(l, view, g_push.frame_ubo->frustum_planes) ? 1u : 0u;
}


// -----------------------------------------------------------------------------
// Pass 2: local wave-prefix scan per group, outputs inclusive prefix[] and
// per-group sums in group_sums[].
// -----------------------------------------------------------------------------
groupshared uint g_wave_sums[MAX_WAVES_PER_GROUP];

[numthreads(THREADS_PER_GROUP, 1, 1)]
[shader("compute")]
void LightScanLocalCS(uint3 dtid : SV_DispatchThreadID,
                      uint3 gid  : SV_GroupID,
                      uint  gidx : SV_GroupIndex)
{
    uint global_idx = dtid.x;

    uint v = 0u;
    if (global_idx < g_push.light_count)
        v = g_push.flags[global_idx];

    uint lane      = WaveGetLaneIndex();
    uint wave_size = WaveGetLaneCount();
    uint wave_id   = gidx / wave_size;

    uint wave_prefix = WavePrefixSum(v);

    uint wave_count = (THREADS_PER_GROUP + wave_size - 1u) / wave_size;
    if (wave_id < MAX_WAVES_PER_GROUP && lane == wave_size - 1u)
        g_wave_sums[wave_id] = wave_prefix + v; // inclusive sum

    GroupMemoryBarrierWithGroupSync();

    if (gidx < wave_count)
    {
        uint sum = g_wave_sums[gidx];
        uint acc = 0u;
        [unroll]
        for (uint k = 0u; k < gidx; ++k)
            acc += g_wave_sums[k];
        g_wave_sums[gidx] = acc + sum;
    }

    GroupMemoryBarrierWithGroupSync();

    uint wave_offset = 0u;
    if (wave_id > 0u && wave_id < MAX_WAVES_PER_GROUP)
        wave_offset = g_wave_sums[wave_id - 1u];

    uint inclusive = wave_prefix + wave_offset + v;

    if (global_idx < g_push.light_count)
        g_push.prefix[global_idx] = inclusive;

    uint last_in_group = (gid.x * THREADS_PER_GROUP + THREADS_PER_GROUP - 1u);
    if (global_idx == last_in_group ||
        (global_idx == g_push.light_count - 1u))
    {
        if (global_idx < g_push.light_count)
            g_push.group_sums[gid.x] = inclusive;
    }
}

// -----------------------------------------------------------------------------
// Pass 3: scan group_sums[0..group_count-1] into exclusive group_offsets[]
// -----------------------------------------------------------------------------
[numthreads(THREADS_PER_GROUP, 1, 1)]
[shader("compute")]
void LightScanGroupsCS(uint3 dtid : SV_DispatchThreadID)
{
    uint idx = dtid.x;
    if (idx >= g_push.group_count)
        return;

    uint acc = 0u;
    [unroll]
    for (uint k = 0u; k < idx; ++k)
        acc += g_push.group_sums[k];

    g_push.group_offsets[idx] = acc;
}

// -----------------------------------------------------------------------------
// Pass 4: compaction – write surviving lights into compact[]
// KEY FIX: Only threads with flag=1 participate, using prefix[i]-1 as index
// -----------------------------------------------------------------------------
[numthreads(THREADS_PER_GROUP, 1, 1)]
[shader("compute")]
void LightCompactCS(uint3 dtid : SV_DispatchThreadID,
                    uint3 gid  : SV_GroupID,
                    uint  gidx : SV_GroupIndex)
{
    uint global_idx = dtid.x;
    if (global_idx >= g_push.light_count)
        return;

    uint flag = g_push.flags[global_idx];
    if (flag == 0u)
        return;

    uint local_inclusive = g_push.prefix[global_idx];
    uint base_offset     = g_push.group_offsets[gid.x];

    uint dst = base_offset + local_inclusive - 1u;

    PointLight l = g_push.lights[global_idx];
    g_push.compact[dst].position_radius = l.position_radius;
    g_push.compact[dst].colour_intensity = l.colour_intensity;

}

[shader("compute")]
[numthreads(1, 1, 1)]
void SetupIndirectCS()
{
    uint last_group    = g_push.group_count - 1u;
    uint last_offset   = g_push.group_offsets[last_group];
    uint last_sum      = g_push.group_sums[last_group];
    uint total_compact = last_offset + last_sum;

    g_push.indirect->index_count    = 36;            // Cube indices
    g_push.indirect->instance_count = total_compact; // Our culled light count
    g_push.indirect->first_index    = 0;
    g_push.indirect->vertex_offset  = 0;
    g_push.indirect->first_instance = 0;

    g_push.indirect_mesh->wgx =(total_compact + 15u) / 16u;
    g_push.indirect_mesh->wgy = 1;
    g_push.indirect_mesh->wgz = 1;
}