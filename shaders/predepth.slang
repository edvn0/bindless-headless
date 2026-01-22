import common;
import maths;

struct TaskData {
    uint base_cube_idx;
};

struct PushConstants
{
    UBO*             frame_ubo;
    Cube* cubes;
    IndirectMeshCommand* indirect_mesh;
};

groupshared TaskData payload;

[shader("amplification")]
[numthreads(16, 1, 1)]
void main_ts(uint gtid : SV_GroupThreadID, uint gid : SV_GroupID, uniform PushConstants g_push) {
    if (gtid == 0) {
        payload.base_cube_idx = gid * 16;
    }
    GroupMemoryBarrierWithGroupSync();

    uint first_cube = payload.base_cube_idx;
    uint count = 16;
    if (first_cube + count > g_push.indirect_mesh->wgx * 16)
        count = g_push.indirect_mesh->wgx * 16 - first_cube;

    if (count > 0 && gtid == 0) {
        DispatchMesh(1, 1, 1, payload);
    }
}

[shader("mesh")]
[numthreads(16, 1, 1)]
[outputtopology("triangle")]
void main_ms(
    uint gtid : SV_GroupThreadID,
    in payload TaskData payload,
    out vertices float4 out_pos[128] : SV_Position,
    out indices uint3 prims[192], 
    uniform PushConstants g_push
) {
    uint cube_idx = payload.base_cube_idx + gtid;

    if (cube_idx >= g_push.indirect_mesh->wgx * 16)
        return;

    Cube cube = g_push.cubes[cube_idx];
    float3 center = cube.pos_radius.xyz;
    float radius = cube.pos_radius.w;

    // Each thread writes 8 vertices
    uint v_base = gtid * 8;
    uint t_base = gtid * 12;

if (gtid == 0)
{
    SetMeshOutputCounts(128, 192);
}
GroupMemoryBarrierWithGroupSync();

    var vp = g_push.frame_ubo->view_projection;

    [unroll]
    for (uint i = 0; i < 8; i++) {
        float3 unit_pos = float3((i & 1) << 1, (i & 2), (i >> 1) & 2) - 1.0;
        float3 world_pos = unit_pos * radius + center;
        out_pos[v_base + i] = mul(vp, float4(world_pos, 1.0));
    }

    uint3 cube_indices[12] = {
        uint3(0,2,1), uint3(1,2,3), uint3(4,5,6), uint3(5,7,6),
        uint3(0,4,2), uint3(2,4,6), uint3(1,3,5), uint3(3,7,5),
        uint3(0,1,4), uint3(1,5,4), uint3(2,6,3), uint3(3,6,7)
    };
    [unroll]
    for (uint j = 0; j < 12; j++) {
        prims[t_base + j] = cube_indices[j] + v_base;
    }
}

struct VertexOutput {
    float4 position;
};

struct VertexInput {
    float3 position;
};

struct PC {
    UBO* frame_ubo;
    float4x4* transforms;
};
[shader("vertex")]
float4 main_vs(VertexInput input,  uint instance: SV_InstanceID, uniform PC pc): SV_Position {
    VertexOutput output = {};
    return mul(pc.frame_ubo->view_projection, mul(pc.transforms[instance] ,float4(input.position, 1.0)));
}

[shader("fragment")]
void main_fs() {
    
}
