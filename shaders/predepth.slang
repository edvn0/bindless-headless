import common;
import maths;

struct TaskData {
    uint base_light_idx;
    uint lod_mask;
};

struct PushConstants
{
    UBO*             frame_ubo;
    PointLight*      compact;
    IndirectCommand* indirect;
    IndirectMeshCommand* indirect_mesh;
};

[[vk::push_constant]]
PushConstants g_push;

groupshared TaskData payload;

[shader("amplification")]
[numthreads(16, 1, 1)]
void main_ts(uint gtid : SV_GroupThreadID, uint gid : SV_GroupID) {
    if (gtid == 0) {
        payload.base_light_idx = gid * 16;
        payload.lod_mask = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint light_idx = payload.base_light_idx + gtid;
    uint lod = 0;

    if (light_idx < g_push.indirect->instance_count) {
        PointLight light = g_push.compact[light_idx];
        // For Pre-Depth, we usually only care about high-detail (Cubes)
        // or we cull very small lights entirely to save Z-buffer bandwidth.
        if (sphere_in_frustum(light.position_radius.xyz, light.position_radius.w, g_push.frame_ubo->frustum_planes)) {
            lod = 3; // Everything is a cube for the Z-Buffer
        }
    }

    uint group_lod_mask = WaveActiveBitOr(lod << (gtid * 2));

    if (gtid == 0) {
        payload.lod_mask = group_lod_mask;
        if (group_lod_mask != 0) DispatchMesh(1, 1, 1, payload);
        else DispatchMesh(0, 0, 0, payload);
    }
}

[shader("mesh")]
[numthreads(16, 1, 1)]
[outputtopology("triangle")]
void main_ms(
    uint gtid : SV_GroupThreadID,
    in payload TaskData payload,
    out vertices float4 out_pos[128], // Only output Position!
    out indices uint3 prims[192]
) {
    uint light_idx = payload.base_light_idx + gtid;
    uint lod = (payload.lod_mask >> (gtid * 2)) & 0x3;

    uint v_count = (lod == 3) ? 8 : 0;
    uint t_count = (lod == 3) ? 12 : 0;

    uint v_base = WavePrefixSum(v_count);
    uint t_base = WavePrefixSum(t_count);
    SetMeshOutputCounts(WaveActiveSum(v_count), WaveActiveSum(t_count));

    if (lod < 3) return;

    PointLight light = g_push.compact[light_idx];

    // Position only - No colors, no normals, no world_pos texcoords
    [unroll]
    for (uint i = 0; i < 8; i++) {
        float3 unit_pos = float3((i & 1) << 1, (i & 2), (i >> 1) & 2) - 1.0;
        float3 world_pos = unit_pos * light.position_radius.w + light.position_radius.xyz;
        out_pos[v_base + i] = mul(g_push.frame_ubo->projection,
                              mul(g_push.frame_ubo->view, float4(world_pos, 1.0)));
    }

    uint3 cube_indices[12] = {
        uint3(0, 2, 1), uint3(1, 2, 3), uint3(4, 5, 6), uint3(5, 7, 6),
        uint3(0, 4, 2), uint3(2, 4, 6), uint3(1, 3, 5), uint3(3, 7, 5),
        uint3(0, 1, 4), uint3(1, 5, 4), uint3(2, 6, 3), uint3(3, 6, 7)
    };
    [unroll]
    for (uint j = 0; j < 12; j++) {
        prims[t_base + j] = cube_indices[j] + v_base;
    }
}