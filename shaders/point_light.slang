import common;
import maths;

struct PushConstants
{
    UBO* frame_ubo;
    Cube* cubes;
    IndirectMeshCommand* indirect_mesh;
};

[[vk::push_constant]]
PushConstants g_push;

struct VertexOutput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
    float3 world_pos : TEXCOORD0;
};

struct TaskData {
    uint base_cube_idx;
};

groupshared TaskData payload;

// -----------------------------------------------------------------------------
// Task Shader: just sets up base indices
// -----------------------------------------------------------------------------
[shader("amplification")]
[numthreads(16, 1, 1)]
void main_ts(uint gtid : SV_GroupThreadID, uint gid : SV_GroupID) {
    if (gtid == 0) {
        payload.base_cube_idx = gid * 16;
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    uint first_cube = payload.base_cube_idx;
    uint total_cubes = g_push.indirect_mesh->wgx * 16;
    uint count = min(16, total_cubes - first_cube);
    
    if (count > 0 && gtid == 0) {
        DispatchMesh(1, 1, 1, payload);
    }
}

// -----------------------------------------------------------------------------
// Mesh Shader: generate cube geometry for each cube
// -----------------------------------------------------------------------------
[shader("mesh")]
[numthreads(16, 1, 1)]
[outputtopology("triangle")]
void main_ms(
    uint gtid : SV_GroupThreadID,
    in payload TaskData payload,
    out vertices VertexOutput verts[128],   // 16 cubes * 8 vertices
    out indices uint3 prims[192]            // 16 cubes * 12 triangles
) {
    uint cube_idx = payload.base_cube_idx + gtid;
    uint total_cubes = g_push.indirect_mesh->wgx * 16;
    
    // CRITICAL FIX: Output counts for ALL threads, not per-thread
    // 16 threads * 8 verts = 128, 16 threads * 12 tris = 192
    SetMeshOutputCounts(8, 12);
    
    // Bounds check
    if (cube_idx >= total_cubes) {
        return;
    }
    
    Cube cube = g_push.cubes[cube_idx];
    float3 center = cube.pos_radius.xyz;
    float radius = cube.pos_radius.w;
    float3 color = cube.colour_intensity.xyz;
    var vp = g_push->frame_ubo->view_projection;
    
    // --- Cube Vertices ---
    uint v_base = gtid * 8;
    [unroll]
    for (uint i = 0; i < 8; i++) {
        // Generate cube corner positions: (-1,-1,-1) to (1,1,1)
        float3 unit_pos = float3((i & 1) * 2 - 1, 
                                 ((i >> 1) & 1) * 2 - 1, 
                                 ((i >> 2) & 1) * 2 - 1);
        float3 world_pos = unit_pos * radius + center;
        
        var v = &verts[v_base + i];
        v->world_pos = world_pos;
        v->color = color;
        v->position = mul(vp, float4(world_pos, 1.0));
    }
    
    // --- Cube Indices ---
    // Standard cube topology
    uint3 cube_indices[12] = {
        uint3(0, 2, 1), uint3(1, 2, 3), // -Z face
        uint3(4, 5, 6), uint3(5, 7, 6), // +Z face
        uint3(0, 4, 2), uint3(2, 4, 6), // -X face
        uint3(1, 3, 5), uint3(3, 7, 5), // +X face
        uint3(0, 1, 4), uint3(1, 5, 4), // -Y face
        uint3(2, 6, 3), uint3(3, 6, 7)  // +Y face
    };
    
    uint t_base = gtid * 12;
    [unroll]
    for (uint j = 0; j < 12; j++) {
        prims[t_base + j] = cube_indices[j] + v_base;
    }
}

[shader("fragment")]
float4 main_fs(VertexOutput input) : SV_Target0
{
    float3 color = input.color;
    float3 light_dir = normalize(float3(1.0, 1.0, -1.0));
    float3 normal = normalize(cross(ddx(input.world_pos), ddy(input.world_pos)));
    float ndotl = max(0.2, dot(normal, light_dir));
    color *= ndotl;
    return float4(color, 1.0);
}