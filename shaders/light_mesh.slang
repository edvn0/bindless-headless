
import common;
import maths;
import pointer;

struct DebugClusteredPushConstants {
    DeviceReadPtr<UBO>        frame_ubo;
    DeviceReadPtr<PointLight> all_lights;
    DeviceReadPtr<Cluster>    clusters;
    DeviceReadPtr<uint>       cluster_light_indices;
    uint tiles_x;
    uint tiles_y;
    uint tiles_z;
};

struct TaskPayload {
    uint cluster_idx;
    uint base_light_offset;
};

struct DebugVertexOut {
    float4 pos : SV_Position;
    float4 color : COLOR0;
};

[shader("amplification")]
[numthreads(1, 1, 1)] 
void main_as(
    uint gid : SV_GroupID,
    uniform DebugClusteredPushConstants pc)
{
    uint cluster_idx = gid;
    Cluster cluster = pc.clusters[cluster_idx];
    uint light_count = cluster.light_count;

    if (light_count > 0) {
    payload TaskPayload p;
        p.cluster_idx = cluster_idx;
        p.base_light_offset = cluster.light_offset;

        // Spawn one Mesh Shader group for every light in this cluster
        // This bypasses the 8-light limit per mesh shader
        DispatchMesh(light_count, 1, 1, p);
    }
}

[shader("mesh")]
[numthreads(32, 1, 1)] // 32 threads is plenty for one cube
[outputtopology("triangle")]
void main_ms(
    uint tid : SV_GroupIndex,
    uint gid : SV_GroupID, 
    in payload TaskPayload payload,
    uniform DebugClusteredPushConstants pc,
    out vertices DebugVertexOut out_vertices[8],
    out indices uint3 out_indices[12])
{
    // Each group handles exactly one light from the cluster's list
    uint light_entry_idx = payload.base_light_offset + gid;
    uint light_idx = pc.cluster_light_indices[light_entry_idx];
    PointLight light = pc.all_lights[light_idx];

    SetMeshOutputCounts(8, 12);

    if (tid < 8) {
        float3 local_pos = float3(
            ((tid & 1) == 0) ? 1.0 : -1.0,
            ((tid & 2) == 0) ? 1.0 : -1.0,
            ((tid & 4) == 0) ? 1.0 : -1.0
        );

        float3 world_pos = (local_pos * light.position_radius.w * 0.05F) + light.position_radius.xyz;
        out_vertices[tid].pos = mul(pc.frame_ubo->view_projection, float4(world_pos, 1.0));
        
        float3 debug_col = float3(
    float(light_idx & 0xFF) / 255.0,
    float((light_idx >> 8) & 0xFF) / 255.0,
    float((light_idx >> 16) & 0xFF) / 255.0
);
        
        out_vertices[tid].color = float4(debug_col, 1.0);
    }

    if (tid < 12) {
        static const uint3 cube_indices[12] = {
            uint3(0, 2, 1), uint3(1, 2, 3), uint3(4, 5, 6), uint3(5, 7, 6),
            uint3(0, 1, 4), uint3(1, 5, 4), uint3(2, 6, 3), uint3(3, 6, 7),
            uint3(0, 4, 2), uint3(2, 4, 6), uint3(1, 3, 5), uint3(3, 7, 5)
        };
        out_indices[tid] = cube_indices[tid];
    }
}

[shader("fragment")]
float4 main_fs_debug(DebugVertexOut input) : SV_Target0
{
    // 'input.color' is now the color for this specific pixel
    return input.color; 
}