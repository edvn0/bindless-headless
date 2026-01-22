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
    float4 color;       // padded
    float4 world_pos;   // padded
};

struct TaskData {
    uint base_cube_idx;
    uint cubes_in_group;  // number of cubes this mesh threadgroup will process
};

groupshared TaskData payload;

// -----------------------------------------------------------------------------
// Task Shader
// -----------------------------------------------------------------------------
[shader("amplification")]
[numthreads(8,1,1)]
void main_ts(uint gtid : SV_GroupThreadID, uint gid : SV_GroupID)
{
    if (gtid == 0)
    {
        uint total_cubes = g_push.indirect_mesh->wgx * 8;
        uint base_idx = gid * 8;
        uint cubes_this_group = min(8, total_cubes - base_idx);

        payload.base_cube_idx = base_idx;
        payload.cubes_in_group = cubes_this_group;
    }

    GroupMemoryBarrierWithGroupSync();

    if (gtid == 0 && payload.cubes_in_group > 0)
    {
        DispatchMesh(1,1,1,payload);
    }
}

// -----------------------------------------------------------------------------
// Mesh Shader
// -----------------------------------------------------------------------------
[shader("mesh")]
[numthreads(8,1,1)]
[outputtopology("triangle")]
void main_ms(
    uint gtid : SV_GroupThreadID,
    in payload TaskData payload,
    out vertices VertexOutput verts[8*8],     // max 8 cubes * 8 vertices
    out indices uint3 prims[8*12]             // max 8 cubes * 12 triangles
)
{
    // Only threads corresponding to actual cubes emit vertices
    if (gtid >= payload.cubes_in_group)
    {
        SetMeshOutputCounts(0,0);
        return;
    }

    SetMeshOutputCounts(8,12); // 1 cube per thread

    Cube cube = g_push.cubes[payload.base_cube_idx + gtid];
    float3 center = cube.pos_radius.xyz;
    float radius = cube.pos_radius.w;
    float3 color = cube.colour_intensity.xyz;
    var vp = g_push->frame_ubo->view_projection;

    // Local per-thread vertex array
    VertexOutput local_verts[8];

    // Generate cube vertices
    [unroll]
    for (uint i=0; i<8; i++)
    {
        float3 unit_pos = float3((i & 1)*2-1,
                                 ((i>>1)&1)*2-1,
                                 ((i>>2)&1)*2-1);
        float3 world_pos = unit_pos * radius + center;

        local_verts[i].world_pos = float4(world_pos,1.0);
        local_verts[i].color     = float4(color,0.0);
        local_verts[i].position  = mul(vp,float4(world_pos,1.0));
    }

    // Standard cube indices
    uint3 cube_indices[12] = {
        uint3(0,2,1), uint3(1,2,3),
        uint3(4,5,6), uint3(5,7,6),
        uint3(0,4,2), uint3(2,4,6),
        uint3(1,3,5), uint3(3,7,5),
        uint3(0,1,4), uint3(1,5,4),
        uint3(2,6,3), uint3(3,6,7)
    };

    // Emit vertices and primitives
    for (uint i=0;i<8;i++)
        verts[i] = local_verts[i]; // thread-local -> mesh output

    for (uint j=0;j<12;j++)
        prims[j] = cube_indices[j];
}

struct VertexInput {
    float3 position;
};

struct PC {
    UBO* frame_ubo;
    float4x4* transforms;
};
[shader("vertex")]
VertexOutput main_vs(VertexInput input,  uint instance: SV_InstanceID, uniform PC pc) {
    VertexOutput output = {};
    output.position=mul(pc.frame_ubo->view_projection, mul(pc.transforms[instance] ,float4(input.position, 1.0)));
    output.world_pos = mul(pc.transforms[instance] ,float4(input.position, 1.0));
    output.color = float4(0.1, 0.9, 0.1, 1.0);
    return output;
}

// -----------------------------------------------------------------------------
// Fragment Shader
// -----------------------------------------------------------------------------
[shader("fragment")]
float4 main_fs(VertexOutput input) : SV_Target0
{
    float3 color = input.color.xyz;
    float3 light_dir = normalize(float3(1.0,1.0,-1.0));
    float3 normal = normalize(cross(ddx(input.world_pos.xyz),ddy(input.world_pos.xyz)));
    float ndotl = max(0.2,dot(normal,light_dir));
    color *= ndotl;
    return float4(color,1.0);
}
