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

struct ClusteredLightCullingPushConstants {
    DeviceReadPtr<UBO> frame_ubo;
    DeviceReadPtr<PointLight> all_lights; // full unculled array
    DeviceWritePtr<VkDrawMeshTasksIndirectCommandEXT> indirect_command;
    DeviceWritePtr<Cluster> clusters;
    DeviceWritePtr<uint> cluster_light_indices;

    float z_near;
    float z_far;
    float log_z_scale;

    uint light_count; 
    uint tiles_x;
    uint tiles_y;
    uint tiles_z;
    uint cluster_count;
};

bool light_intersects_cluster(ClusteredLightCullingPushConstants pc,
                               uint light_index, uint cx, uint cy, uint cz) {
    var l = pc.all_lights[light_index];
    float3 view_pos = mul(pc.frame_ubo->view, float4(l.position_radius.xyz, 1.0)).xyz;
    float radius = l.position_radius.w;

    float z_near_slice = pc.z_near * pow(2.0, float(cz)     / pc.log_z_scale);
    float z_far_slice  = pc.z_near * pow(2.0, float(cz + 1) / pc.log_z_scale);

    float pos_z = -view_pos.z;
    if (pos_z + radius < z_near_slice || pos_z - radius > z_far_slice)
        return false;

float2 ndc_min = float2(
     (float(cx)     / pc.tiles_x) * 2.0 - 1.0,
    -(float(cy + 1) / pc.tiles_y) * 2.0 + 1.0  
);
float2 ndc_max = float2(
     (float(cx + 1) / pc.tiles_x) * 2.0 - 1.0,
    -(float(cy)     / pc.tiles_y) * 2.0 + 1.0 
);

float4x4 inv_proj = pc.frame_ubo->inv_projection;
float3 tl = unproject_to_view(inv_proj, float2(ndc_min.x, ndc_max.y), 1.0);
float3 tr = unproject_to_view(inv_proj, float2(ndc_max.x, ndc_max.y), 1.0);
float3 bl = unproject_to_view(inv_proj, float2(ndc_min.x, ndc_min.y), 1.0);
float3 br = unproject_to_view(inv_proj, float2(ndc_max.x, ndc_min.y), 1.0);

float3 center = unproject_to_view(inv_proj, (ndc_min + ndc_max) * 0.5, 1.0);

float3 planes[4];
planes[0] = cross(tl, bl);
planes[1] = cross(br, tr);
planes[2] = cross(bl, br);
planes[3] = cross(tr, tl);

for (int i = 0; i < 4; ++i) {
    if (dot(planes[i], center) < 0.0)
        planes[i] = -planes[i];
}

for (int i = 0; i < 4; ++i) {
    float3 n = safe_normalize(planes[i]);
    if (dot(n, view_pos) < -radius)
        return false;
}

    return true;
}

groupshared uint g_light_count;
groupshared uint g_light_indices[MAX_LIGHTS_PER_CLUSTER];
groupshared float4 g_planes[4]; 
groupshared float g_z_near;
groupshared float g_z_far;
groupshared float4x4 g_view_mat;

[numthreads(THREADS_PER_GROUP, 1, 1)][shader("compute")] 
void BuildClusterCS(
    uint3 group_id : SV_GroupID, 
    uint group_index : SV_GroupIndex,
    uniform ClusteredLightCullingPushConstants pc) 
{
    uint cluster_idx = group_id.x;
    if (cluster_idx >= pc.cluster_count) return;

    if (group_index == 0) {
        g_light_count = 0;
        g_view_mat = pc.frame_ubo->view;

        uint cz = cluster_idx / (pc.tiles_x * pc.tiles_y);
        uint cy = (cluster_idx % (pc.tiles_x * pc.tiles_y)) / pc.tiles_x;
        uint cx = cluster_idx % pc.tiles_x;

        g_z_near = pc.z_near * exp2(float(cz) / pc.log_z_scale);
        g_z_far  = pc.z_near * exp2(float(cz + 1) / pc.log_z_scale);

        float2 ndc_min = float2((float(cx)/pc.tiles_x)*2.0-1.0, -(float(cy+1)/pc.tiles_y)*2.0+1.0);
        float2 ndc_max = float2((float(cx+1)/pc.tiles_x)*2.0-1.0, -(float(cy)/pc.tiles_y)*2.0+1.0);
        
        float4x4 inv_proj = pc.frame_ubo->inv_projection;
        float3 tl = unproject_to_view(inv_proj, float2(ndc_min.x, ndc_max.y), 1.0);
        float3 tr = unproject_to_view(inv_proj, float2(ndc_max.x, ndc_max.y), 1.0);
        float3 bl = unproject_to_view(inv_proj, float2(ndc_min.x, ndc_min.y), 1.0);
        float3 br = unproject_to_view(inv_proj, float2(ndc_max.x, ndc_min.y), 1.0);

        g_planes[0] = float4(safe_normalize(cross(tl, bl)), 0.0);
        g_planes[1] = float4(safe_normalize(cross(br, tr)), 0.0);
        g_planes[2] = float4(safe_normalize(cross(bl, br)), 0.0);
        g_planes[3] = float4(safe_normalize(cross(tr, tl)), 0.0);
        
        float3 center = unproject_to_view(inv_proj, (ndc_min + ndc_max) * 0.5, 1.0);
        for(int i=0; i<4; ++i) {
            if(dot(g_planes[i].xyz, center) < 0.0) g_planes[i] *= -1.0;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    for (uint i = group_index; i < pc.light_count; i += THREADS_PER_GROUP) {
        if (WaveReadLaneFirst(g_light_count) >= (uint)MAX_LIGHTS_PER_CLUSTER)
            break;
        var l = pc.all_lights[i];
        float4 pos_rad = l.position_radius;
        float3 view_pos = mul(g_view_mat, float4(pos_rad.xyz, 1.0)).xyz;
        float radius = pos_rad.w;

        bool hit = true;

        float pos_z = -view_pos.z;
        if (pos_z + radius < g_z_near || pos_z - radius > g_z_far) {
            hit = false;
        }

    if (WaveActiveAnyTrue(hit)) {
        [unroll]
        for (int p = 0; p < 4 && hit; ++p) {
            if (dot(g_planes[p].xyz, view_pos) < -radius)
                hit = false;
        }
    }

        uint hit_count = WaveActiveCountBits(hit);
        uint prefix_hits = WavePrefixCountBits(hit);
        uint wave_offset = 0u;

        if (WaveIsFirstLane() && hit_count > 0u)
            InterlockedAdd(g_light_count, hit_count, wave_offset);

        wave_offset = WaveReadLaneFirst(wave_offset);

        if (hit) {
            uint slot = wave_offset + prefix_hits;
            if (slot < MAX_LIGHTS_PER_CLUSTER)
                g_light_indices[slot] = i;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    uint final_count = min(g_light_count, (uint)MAX_LIGHTS_PER_CLUSTER);
    uint base_offset = cluster_idx * MAX_LIGHTS_PER_CLUSTER;

    if (group_index == 0) {
        pc.clusters[cluster_idx].light_offset = base_offset;
        pc.clusters[cluster_idx].light_count = final_count;
    }

    for (uint j = group_index; j < final_count; j += THREADS_PER_GROUP)
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