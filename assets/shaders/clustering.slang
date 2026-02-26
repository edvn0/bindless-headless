import common;
import maths;
import pointer;
import constants;


[[vk::constant_id(0)]]
const int MAX_WAVES_PER_GROUP = 4;
[[vk::constant_id(1)]]
const int THREADS_PER_GROUP = 64;

struct VkDrawMeshTasksIndirectCommandEXT {
    uint groupCountX;
    uint groupCountY;
    uint groupCountZ;
};

struct ClusteredLightCullingPushConstants
{
    DeviceReadPtr<UBO>        frame_ubo;
    DeviceReadPtr<PointLight> all_lights;       // full unculled array
    DeviceWritePtr<VkDrawMeshTasksIndirectCommandEXT> indirect_command;
    DeviceWritePtr<Cluster> clusters;
    DeviceWritePtr<uint>    cluster_light_indices;
    
    float z_near;
    float z_far;
    float log_z_scale;
    
    uint                      light_count;  // full unculled count (plain uint, no pointer)
    uint tiles_x;
    uint tiles_y;
    uint tiles_z;
    uint cluster_count;

};

bool light_intersects_cluster(ClusteredLightCullingPushConstants pc, uint light_index, uint cx, uint cy, uint cz)
{
    var l = pc.all_lights[light_index];
    float3 view_pos = mul(pc.frame_ubo->view, float4(l.position_radius.xyz, 1.0)).xyz;
    float  radius   = l.position_radius.w;

    // Frustum cull first — cheap reject before AABB test
//if (!sphere_in_frustum(view_pos, radius, pc.frame_ubo->frustum_planes))
//        return false;
    float view_z = -view_pos.z;
if (view_z + radius < pc.z_near || view_z - radius > pc.z_far)
    return false;

    var view_data = ClusterViewData(
        pc.tiles_x, pc.tiles_y, pc.tiles_z,
        pc.log_z_scale, pc.z_near, pc.z_far,
        pc.frame_ubo->inv_projection);

    ViewAABB box = get_cluster_view_aabb(cx, cy, cz, view_data);

    float sq_dist = 0.0;
    for (int i = 0; i < 3; ++i) {
        float v = view_pos[i];
        if (v < box.min_v[i]) sq_dist += (box.min_v[i] - v) * (box.min_v[i] - v);
        if (v > box.max_v[i]) sq_dist += (v - box.max_v[i]) * (v - box.max_v[i]);
    }

    return sq_dist < (radius * radius);
}

groupshared uint g_light_count;
groupshared uint g_light_indices[MAX_LIGHTS_PER_CLUSTER];

[numthreads(CLUSTER_BUILD_THREADS, 1, 1)]
[shader("compute")]
void BuildClusterCS(
    uint3 group_id    : SV_GroupID,
    uint  group_index : SV_GroupIndex,
    uniform ClusteredLightCullingPushConstants pc)
{
    uint cluster_idx = group_id.x;
    if (cluster_idx >= pc.cluster_count) return;

    uint z = cluster_idx / (pc.tiles_x * pc.tiles_y);
    uint y = (cluster_idx % (pc.tiles_x * pc.tiles_y)) / pc.tiles_x;
    uint x = cluster_idx % pc.tiles_x;

    if (group_index == 0) g_light_count = 0;
    GroupMemoryBarrierWithGroupSync();

    uint wave_size = WaveGetLaneCount();

    for (uint i = group_index; i < pc.light_count; i += CLUSTER_BUILD_THREADS)
    {
        bool hit = light_intersects_cluster(pc, i, x, y, z);

        uint hit_count   = WaveActiveCountBits(hit);
        uint prefix_hits = WavePrefixCountBits(hit);

        uint wave_offset = 0u;
        if (WaveIsFirstLane() && hit_count > 0u)
            InterlockedAdd(g_light_count, hit_count, wave_offset);

        wave_offset = WaveReadLaneFirst(wave_offset);

        if (hit)
        {
            uint slot = wave_offset + prefix_hits;
            if (slot < MAX_LIGHTS_PER_CLUSTER)
                g_light_indices[slot] = i;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    uint final_count = min(g_light_count, MAX_LIGHTS_PER_CLUSTER);
    uint base_offset = cluster_idx * MAX_LIGHTS_PER_CLUSTER;

    if (group_index == 0)
    {
        pc.clusters[cluster_idx].light_offset = base_offset;
        pc.clusters[cluster_idx].light_count  = final_count;
    }

    for (uint j = group_index; j < final_count; j += CLUSTER_BUILD_THREADS)
        pc.cluster_light_indices[base_offset + j] = g_light_indices[j];
}

[numthreads(1, 1, 1)]
[shader("compute")]
void LightFinaliseCS(uniform ClusteredLightCullingPushConstants pc)
{
    pc.indirect_command[0].groupCountX = pc.light_count;
    pc.indirect_command[0].groupCountY = 1;
    pc.indirect_command[0].groupCountZ = 1;
}