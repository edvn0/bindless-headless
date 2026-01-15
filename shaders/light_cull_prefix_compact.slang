// shaders/light_cull_prefix_compact.slang
// Pointer-based typed access with device addresses passed via push constants.

import common;

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
    uint        image_index;
    uint        light_count;
    uint        group_count;
};

[[vk::push_constant]]
PushConstants g_push;

bool sphere_in_frustum(float3 view_pos, float radius, float4 frustum_planes[6])
{
    for (int i = 0; i < 6; ++i) {
        float3 normal = frustum_planes[i].xyz;
        float d = frustum_planes[i].w;

        float dist = dot(normal, view_pos) + d;

        if (dist < -radius)
            return false;
    }

    return true;
}

bool light_in_frustum(PointLight l, float4x4 view, float4 frustum_planes[6])
{
    float3 world_pos = l.position_radius.xyz;
    float  radius    = l.position_radius.w;

    float4 view_pos4 = mul(view, float4(world_pos, 1.0f));
    float3 view_pos = view_pos4.xyz;

    return sphere_in_frustum(view_pos, radius, frustum_planes);
}

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
                    uint3 gid  : SV_GroupID)
{
   uint global_idx = dtid.x;
    if (global_idx >= g_push.light_count)
        return;

    uint flag = g_push.flags[global_idx];
    if (flag == 0u)
        return;

    // Read the inclusive prefix and group offset
    uint local_inclusive = g_push.prefix[global_idx];
    uint base_offset     = g_push.group_offsets[gid.x];

    // Compute destination index (convert inclusive to zero-based)
    uint dst = base_offset + local_inclusive - 1u;

    // Write the light
    PointLight l = g_push.lights[global_idx];
    g_push.compact[dst].position_radius = l.position_radius;
g_push.compact[dst].colour_intensity = l.colour_intensity;
}

// -----------------------------------------------------------------------------
// Pass 5: debug draw – draw one pixel for each compacted light
// -----------------------------------------------------------------------------
[numthreads(THREADS_PER_GROUP, 1, 1)]
[shader("compute")]
void LightDebugDrawCS(uint3 dtid : SV_DispatchThreadID)
{
    uint idx = dtid.x;

    if (g_push.group_count == 0u)
        return;

    uint last_group    = g_push.group_count - 1u;
    uint last_offset   = g_push.group_offsets[last_group];
    uint last_sum      = g_push.group_sums[last_group];
    uint compact_count = last_offset + last_sum;

uint max_compact = min(compact_count, g_push.light_count);
if (idx >= max_compact)
    return;

    uint image_idx = NonUniformResourceIndex(g_push.image_index);
    RWTexture2D<float4> image = storage_images[image_idx];

    uint width, height;
    image.GetDimensions(width, height);

    PointLight l = g_push.compact[idx];
    float3 world_pos = l.position_radius.xyz;

    // Use actual projection matrix - transform world to clip space
    float4x4 view = g_push.frame_ubo->view;
    float4x4 proj = g_push.frame_ubo->projection;

    float4 view_pos = mul(view, float4(world_pos, 1.0f));
    float4 clip_pos = mul(proj, view_pos);

    if (clip_pos.w <= 0.0f)
        return;

    float3 ndc = clip_pos.xyz / clip_pos.w;

    // Convert NDC [-1,1] to UV [0,1]
    float2 uv;
    uv.x = ndc.x * 0.5f + 0.5f;
    uv.y = ndc.y * 0.5f + 0.5f;
    uv = saturate(uv);

    uint2 pixel = uint2(
        uint(uv.x * float(width  - 1)),
        uint(uv.y * float(height - 1))
    );

    float3 color = l.colour_intensity.xyz;
    image[pixel] = float4(color, 1.0f);
}