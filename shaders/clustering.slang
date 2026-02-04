import common;

[[vk::constant_id(0)]]
const int MAX_WAVES_PER_GROUP = 4;
[[vk::constant_id(1)]]
const int THREADS_PER_GROUP = 64;
[[vk::constant_id(2)]]
const uint MAX_CLUSTERS_PER_LIGHT = 32;

struct Cluster
{
    uint light_offset;
    uint light_count;
};

struct PushConstants
{
    UBO*        frame_ubo;
    PointLight* lights;
    uint        light_count;

    float z_near;
    float z_far;
    float log_z_scale;

    uint tiles_x;
    uint tiles_y;
    uint tiles_z;

    uint cluster_count;

    uint*       cluster_counts;         // [cluster_count] - counts per cluster
    Cluster*    clusters;               // [cluster_count] - offset and count
    uint*       cluster_counters;       // [cluster_count] - write counters, reset to 0
    uint*       cluster_light_indices;  // [total_lights_across_clusters]
    uint*       global_index_count;     // [1] - total light indices written
};


float2 project_to_ndc(float3 view_pos, float4x4 proj)
{
    float4 clip = mul(proj, float4(view_pos, 1.0));
    return clip.xy / clip.w; // [-1, 1]
}

uint ndc_to_tile_x(float x_ndc, uint tiles_x)
{
    float u = x_ndc * 0.5f + 0.5f;
    return uint(clamp(u * tiles_x, 0.0f, float(tiles_x - 1)));
}

uint ndc_to_tile_y(float y_ndc, uint tiles_y)
{
    float v = 1.0f - (y_ndc * 0.5f + 0.5f);
    return uint(clamp(v * tiles_y, 0.0f, float(tiles_y - 1)));
}

uint z_to_slice(float view_z, float z_near, float log_z_scale)
{
    float depth_ratio = abs(view_z) / z_near;
    return uint(log2(depth_ratio) * log_z_scale);
}

void compute_light_cluster_range(
    PointLight l,
    UBO* ubo,
    float z_near,
    float log_z_scale,
    uint tiles_x,
    uint tiles_y,
    uint tiles_z,
    out uint x0,
    out uint x1,
    out uint y0,
    out uint y1,
    out uint z0,
    out uint z1)
{
    float3 world_pos = l.position_radius.xyz;
    float  radius    = l.position_radius.w;

    // ---- world → view ----
    float3 view_pos = mul(ubo->view, float4(world_pos, 1.0)).xyz;

    // behind camera → reject early
    if (view_pos.z + radius <= 0.0)
    {
        x0 = y0 = z0 = 1;
        x1 = y1 = z1 = 0;
        return;
    }

    // ---- Z slice range ----
    float z_min = max(view_pos.z - radius, -z_near);
    float z_max = view_pos.z + radius;

    z0 = z_to_slice(z_min, z_near, log_z_scale);
    z1 = z_to_slice(z_max, z_near, log_z_scale);
    
    // Clamp to valid range
    z0 = min(z0, tiles_z - 1);
    z1 = min(z1, tiles_z - 1);

    // ---- screen-space bounds ----
    // Conservative: project sphere as AABB in view space
    float3 vx  = view_pos + float3(radius, 0, 0);
    float3 vxn = view_pos - float3(radius, 0, 0);
    float3 vy  = view_pos + float3(0, radius, 0);
    float3 vyn = view_pos - float3(0, radius, 0);

    float2 ndc0 = project_to_ndc(vxn, ubo->projection);
    float2 ndc1 = project_to_ndc(vx,  ubo->projection);
    float2 ndc2 = project_to_ndc(vyn, ubo->projection);
    float2 ndc3 = project_to_ndc(vy,  ubo->projection);

    float ndc_min_x = min(min(ndc0.x, ndc1.x), min(ndc2.x, ndc3.x));
    float ndc_max_x = max(max(ndc0.x, ndc1.x), max(ndc2.x, ndc3.x));
    float ndc_min_y = min(min(ndc0.y, ndc1.y), min(ndc2.y, ndc3.y));
    float ndc_max_y = max(max(ndc0.y, ndc1.y), max(ndc2.y, ndc3.y));

    x0 = ndc_to_tile_x(ndc_min_x, tiles_x);
    x1 = ndc_to_tile_x(ndc_max_x, tiles_x);
    y0 = ndc_to_tile_y(ndc_max_y, tiles_y);
    y1 = ndc_to_tile_y(ndc_min_y, tiles_y);
}

uint cluster_index(uint x, uint y, uint z, uint tiles_x, uint tiles_y)
{
    return x + y * tiles_x + z * tiles_x * tiles_y;
}

// ============================================================================
// Pass 1: Count lights per cluster
// ============================================================================
// Each thread processes one light and atomically increments the count
// for each cluster it overlaps. Uses wave intrinsics to reduce atomics.
// ============================================================================

[numthreads(THREADS_PER_GROUP, 1, 1)]
[shader("compute")]
void LightClusterCountCS(
    uint3 dtid : SV_DispatchThreadID,
    uniform PushConstants pc)
{
    uint light_idx = dtid.x;
    if (light_idx >= pc.light_count)
        return;

    PointLight l = pc.lights[light_idx];

    // ---- compute cluster range ----
    uint x0, x1, y0, y1, z0, z1;
    compute_light_cluster_range(
        l,
        pc.frame_ubo,
        pc.z_near,
        pc.log_z_scale,
        pc.tiles_x, pc.tiles_y, pc.tiles_z,
        x0, x1, y0, y1, z0, z1);

    // Early out if light doesn't overlap any clusters
    if (x0 > x1 || y0 > y1 || z0 > z1)
        return;

    // ---- iterate overlapped clusters ----
    for (uint z = z0; z <= z1; ++z)
    for (uint y = y0; y <= y1; ++y)
    for (uint x = x0; x <= x1; ++x)
    {
        uint c = cluster_index(x, y, z, pc.tiles_x, pc.tiles_y);

        // Wave-aggregated atomic: all lanes in wave that touch this cluster
        // sum their contributions and do a single atomic add
        uint v = 1u;
        uint wave_sum = WaveActiveSum(v);

        if (WaveIsFirstLane())
            InterlockedAdd(pc.cluster_counts[c], wave_sum);
    }
}

// ============================================================================
// Pass 2: Prefix sum to compute light offsets
// ============================================================================
// Computes exclusive prefix sum of cluster_counts to determine where each
// cluster's light indices should be written in the global index buffer.
// Also computes total number of light indices needed.
// ============================================================================

groupshared uint g_wave_sums[MAX_WAVES_PER_GROUP];
groupshared uint g_group_base;

[numthreads(THREADS_PER_GROUP, 1, 1)]
[shader("compute")]
void ClusterPrefixSumCS(
    uint3 dtid : SV_DispatchThreadID,
    uint3 gid  : SV_GroupID,
    uint  gidx : SV_GroupIndex,
    uniform PushConstants pc)
{
    uint idx = dtid.x;

    // Load value for this cluster
    uint v = 0u;
    if (idx < pc.cluster_count)
        v = pc.cluster_counts[idx];

    uint lane      = WaveGetLaneIndex();
    uint wave_size = WaveGetLaneCount();
    uint wave_id   = gidx / wave_size;

    // Prefix sum within wave
    uint wave_prefix = WavePrefixSum(v);

    // Last lane in each wave writes wave total to shared memory
    if (lane == wave_size - 1)
        g_wave_sums[wave_id] = wave_prefix + v;

    GroupMemoryBarrierWithGroupSync();

    // Compute prefix sum across waves (single thread per wave)
    uint wave_count = (THREADS_PER_GROUP + wave_size - 1) / wave_size;

    if (gidx < wave_count)
    {
        uint acc = 0;
        for (uint i = 0; i < gidx; ++i)
            acc += g_wave_sums[i];

        g_wave_sums[gidx] += acc;
    }

    GroupMemoryBarrierWithGroupSync();

    // Compute final prefix sum for this thread
    uint wave_offset = (wave_id > 0) ? g_wave_sums[wave_id - 1] : 0;
    uint inclusive   = wave_prefix + wave_offset + v;
    uint exclusive   = inclusive - v;

    // Write cluster offset and count
    if (idx < pc.cluster_count)
    {
        pc.clusters[idx].light_offset = exclusive;
        pc.clusters[idx].light_count  = v;
    }

    // First thread in group adds this group's total to global counter
    if (gidx == 0)
    {
        uint group_total = g_wave_sums[wave_count - 1];
        InterlockedAdd(pc.global_index_count[0], group_total, g_group_base);
    }

    GroupMemoryBarrierWithGroupSync();

    // Add group base offset to all clusters in this group
    if (idx < pc.cluster_count && gidx == 0)
    {
        uint num_threads_in_group = min(THREADS_PER_GROUP, pc.cluster_count - gid.x * THREADS_PER_GROUP);
        
        for (uint i = 0; i < num_threads_in_group; ++i)
        {
            uint cluster_idx = gid.x * THREADS_PER_GROUP + i;
            if (cluster_idx < pc.cluster_count)
                pc.clusters[cluster_idx].light_offset += g_group_base;
        }
    }
}

// ============================================================================
// Pass 3: Write light indices to clusters
// ============================================================================
// Each thread processes one light again and writes its index to all
// overlapping clusters at the correct offset. Uses wave intrinsics to
// reduce atomic operations when multiple threads write to same cluster.
// ============================================================================

[numthreads(THREADS_PER_GROUP, 1, 1)]
[shader("compute")]
void LightClusterWriteCS(
    uint3 dtid : SV_DispatchThreadID,
    uniform PushConstants pc)
{
    uint light_idx = dtid.x;
    if (light_idx >= pc.light_count)
        return;

    PointLight l = pc.lights[light_idx];

    // ---- compute cluster range again ----
    uint x0, x1, y0, y1, z0, z1;
    compute_light_cluster_range(
        l,
        pc.frame_ubo,
        pc.z_near,
        pc.log_z_scale,
        pc.tiles_x, pc.tiles_y, pc.tiles_z,
        x0, x1, y0, y1, z0, z1);

    // Early out if light doesn't overlap any clusters
    if (x0 > x1 || y0 > y1 || z0 > z1)
        return;

    // ---- iterate overlapped clusters ----
    for (uint z = z0; z <= z1; ++z)
    for (uint y = y0; y <= y1; ++y)
    for (uint x = x0; x <= x1; ++x)
    {
        uint c = cluster_index(x, y, z, pc.tiles_x, pc.tiles_y);

        // Wave-aggregated atomic for write position
        uint v = 1u;

        uint wave_prefix = WavePrefixSum(v);
        uint wave_total  = WaveActiveSum(v);

        // First lane atomically reserves space for entire wave
        uint base;
        if (WaveIsFirstLane())
            InterlockedAdd(pc.cluster_counters[c], wave_total, base);

        // Broadcast base to all lanes
        base = WaveReadLaneFirst(base);

        // Compute final write destination
        uint dst =
            pc.clusters[c].light_offset +
            base +
            wave_prefix;

        // Write light index
        pc.cluster_light_indices[dst] = light_idx;
    }
}
