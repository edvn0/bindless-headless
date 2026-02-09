import common;

static const uint MAX_LIGHTS_PER_CLUSTER = 128;
static const uint CLUSTER_BUILD_THREADS = 64;

[[vk::constant_id(0)]]
const int MAX_WAVES_PER_GROUP = 4;
[[vk::constant_id(1)]]
const int THREADS_PER_GROUP = 64;

struct Cluster
{
    uint light_offset;
    uint light_count;
};

struct PushConstants
{
    UBO*        frame_ubo;
    PointLight* lights; // Compacted frustum culled
    uint*        light_count; // Compacted frustum culled count

    float z_near;
    float z_far;
    float log_z_scale;

    uint tiles_x;
    uint tiles_y;
    uint tiles_z;
    uint cluster_count;

    Cluster*    clusters;               // [cluster_count] - offset and count
    uint*       cluster_light_indices;  // [total_lights_across_clusters]
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



groupshared uint g_light_count;
groupshared uint g_light_indices[MAX_LIGHTS_PER_CLUSTER];


bool light_intersects_cluster(
    PointLight l,
    UBO* ubo,
    float z_near,
    float log_z_scale,
    uint tiles_x,
    uint tiles_y,
    uint tiles_z,
    uint cluster_x,
    uint cluster_y,
    uint cluster_z)
{
    float3 world_pos = l.position_radius.xyz;
    float radius = l.position_radius.w;
    
    float3 view_pos = mul(ubo->view, float4(world_pos, 1.0)).xyz;
    
    if (view_pos.z + radius <= 0.0)
        return false;
    
    float z_min = max(view_pos.z - radius, -z_near);
    float z_max = view_pos.z + radius;
    
    uint z0 = z_to_slice(z_min, z_near, log_z_scale);
    uint z1 = z_to_slice(z_max, z_near, log_z_scale);
    z0 = min(z0, tiles_z - 1);
    z1 = min(z1, tiles_z - 1);
    
    if (cluster_z < z0 || cluster_z > z1)
        return false;
    
    float3 vx = view_pos + float3(radius, 0, 0);
    float3 vxn = view_pos - float3(radius, 0, 0);
    float3 vy = view_pos + float3(0, radius, 0);
    float3 vyn = view_pos - float3(0, radius, 0);
    
    float2 ndc0 = project_to_ndc(vxn, ubo->projection);
    float2 ndc1 = project_to_ndc(vx, ubo->projection);
    float2 ndc2 = project_to_ndc(vyn, ubo->projection);
    float2 ndc3 = project_to_ndc(vy, ubo->projection);
    
    float ndc_min_x = min(min(ndc0.x, ndc1.x), min(ndc2.x, ndc3.x));
    float ndc_max_x = max(max(ndc0.x, ndc1.x), max(ndc2.x, ndc3.x));
    float ndc_min_y = min(min(ndc0.y, ndc1.y), min(ndc2.y, ndc3.y));
    float ndc_max_y = max(max(ndc0.y, ndc1.y), max(ndc2.y, ndc3.y));
    
    uint x0 = ndc_to_tile_x(ndc_min_x, tiles_x);
    uint x1 = ndc_to_tile_x(ndc_max_x, tiles_x);
    uint y0 = ndc_to_tile_y(ndc_max_y, tiles_y);
    uint y1 = ndc_to_tile_y(ndc_min_y, tiles_y);
    
    return (cluster_x >= x0 && cluster_x <= x1 &&
            cluster_y >= y0 && cluster_y <= y1);
}

[numthreads(CLUSTER_BUILD_THREADS, 1, 1)]
[shader("compute")]
void BuildClusterCS(
    uint3 group_id : SV_GroupID,
    uint group_index : SV_GroupIndex,
    uniform PushConstants pc)
{
    uint cluster_idx = group_id.x;
    if (cluster_idx >= pc.cluster_count)
        return;
    
    uint z = cluster_idx / (pc.tiles_x * pc.tiles_y);
    uint rem = cluster_idx % (pc.tiles_x * pc.tiles_y);
    uint y = rem / pc.tiles_x;
    uint x = rem % pc.tiles_x;
    
    if (group_index == 0)
        g_light_count = 0;
    
    GroupMemoryBarrierWithGroupSync();
    
    uint total_lights = *pc.light_count;
    
    for (uint light_idx = group_index; light_idx < total_lights; light_idx += CLUSTER_BUILD_THREADS)
    {
        PointLight light = pc.lights[light_idx];
        
        if (light_intersects_cluster(light, pc.frame_ubo, pc.z_near, pc.log_z_scale,
                                     pc.tiles_x, pc.tiles_y, pc.tiles_z,
                                     x, y, z))
        {
            uint slot;
            InterlockedAdd(g_light_count, 1, slot);
            
            if (slot < MAX_LIGHTS_PER_CLUSTER)
                g_light_indices[slot] = light_idx;
        }
    }
    
    // ===== BARRIER: Wait for all light tests to complete =====
    GroupMemoryBarrierWithGroupSync();
    
    // ===== PHASE 3: Write cluster metadata =====
    uint actual_count = min(g_light_count, MAX_LIGHTS_PER_CLUSTER);
    
    if (group_index == 0)
    {
        pc.clusters[cluster_idx].light_offset = cluster_idx * MAX_LIGHTS_PER_CLUSTER;
        pc.clusters[cluster_idx].light_count = actual_count;
    }
    
    // ===== PHASE 4: Write light indices to global memory =====
    uint base_offset = cluster_idx * MAX_LIGHTS_PER_CLUSTER;
    
    for (uint i = group_index; i < actual_count; i += CLUSTER_BUILD_THREADS)
    {
        pc.cluster_light_indices[base_offset + i] = g_light_indices[i];
    }
}