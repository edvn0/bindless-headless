import common;
import maths;
import pointer;

static const uint MAX_LIGHTS_PER_CLUSTER = 128;
static const uint CLUSTER_BUILD_THREADS = 64;

[[vk::constant_id(0)]]
const int MAX_WAVES_PER_GROUP = 4;
[[vk::constant_id(1)]]
const int THREADS_PER_GROUP = 64;

struct VkDrawMeshTasksIndirectCommandEXT {
    uint groupCountX;
    uint groupCountY;
    uint groupCountZ;
};

struct PushConstants
{
    DeviceReadPtr<UBO>        frame_ubo;
    DeviceReadPtr<PointLight> lights; // Compacted frustum culled
    DeviceReadPtr<uint>        light_count; // Compacted frustum culled count
    DeviceWritePtr<VkDrawMeshTasksIndirectCommandEXT> indirect_command;

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

bool light_intersects_cluster(PushConstants pc, uint light_index, uint cx, uint cy, uint cz) 
{
    var l = pc.lights[light_index];
    float3 view_pos = mul(pc.frame_ubo->view, float4(l.position_radius.xyz, 1.0)).xyz;
    float radius = l.position_radius.w;

    var view_data = ClusterViewData(pc.tiles_x, pc.tiles_y, pc.tiles_z, pc.log_z_scale,
                                    pc.z_near, pc.z_far, pc.frame_ubo->inv_projection);
    ViewAABB box = get_cluster_view_aabb(cx, cy, cz, view_data);

    float sq_dist = 0.0;
    for (int i = 0; i < 3; ++i) {
        float v = view_pos[i];
        if (v < box.min_v[i]) sq_dist += (box.min_v[i] - v) * (box.min_v[i] - v);
        if (v > box.max_v[i]) sq_dist += (v - box.max_v[i]) * (v - box.max_v[i]);
    }

    return sq_dist <= (radius * radius);
}

groupshared uint g_light_count;
groupshared uint g_light_indices[MAX_LIGHTS_PER_CLUSTER];

[numthreads(CLUSTER_BUILD_THREADS, 1, 1)]
[shader("compute")]
void BuildClusterCS(
    uint3 group_id : SV_GroupID,
    uint group_index : SV_GroupIndex,
    uniform PushConstants pc)
{
    uint cluster_idx = group_id.x;
    if (cluster_idx >= pc.cluster_count) return;

    // Decode 3D cluster ID
    uint z = cluster_idx / (pc.tiles_x * pc.tiles_y);
    uint y = (cluster_idx % (pc.tiles_x * pc.tiles_y)) / pc.tiles_x;
    uint x = cluster_idx % pc.tiles_x;

    if (group_index == 0) g_light_count = 0;
    GroupMemoryBarrierWithGroupSync();

    uint total_compact_lights = *pc.light_count;

    // Threads collaborate to test all frustum-culled lights against this specific cluster
    for (uint i = group_index; i < total_compact_lights; i += CLUSTER_BUILD_THREADS) {
        if (light_intersects_cluster(pc,i, x, y, z)) {
            uint slot;
            InterlockedAdd(g_light_count, 1, slot);
            if (slot < MAX_LIGHTS_PER_CLUSTER) {
                g_light_indices[slot] = i;
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();

    uint final_count = min(g_light_count, MAX_LIGHTS_PER_CLUSTER);
    uint base_offset = cluster_idx * MAX_LIGHTS_PER_CLUSTER;

    if (group_index == 0) {
        pc.clusters[cluster_idx].light_offset = base_offset;
        pc.clusters[cluster_idx].light_count = final_count;
    }

    // Write global light indices
    for (uint j = group_index; j < final_count; j += CLUSTER_BUILD_THREADS) {
        pc.cluster_light_indices[base_offset + j] = g_light_indices[j];
    }
}

[numthreads(1, 1, 1)]
[shader("compute")]
void LightFinaliseCS(uniform PushConstants pc)
{
    uint total_culled = *pc.light_count;
    
    pc.indirect_command[0].groupCountX = total_culled;
    pc.indirect_command[0].groupCountY = 1;
    pc.indirect_command[0].groupCountZ = 1;
}
