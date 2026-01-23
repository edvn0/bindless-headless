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
    float4 position  : SV_Position;
    float3 world_pos : TEXCOORD0;
    float3 normal    : TEXCOORD1;
    float2 uv        : TEXCOORD2;
    float3 color     : TEXCOORD3;
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

        local_verts[i].world_pos = world_pos;
        local_verts[i].color     = color;
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
    uint normal;
    uint uvs;
};

struct PC {
    UBO* frame_ubo;
    float4x4* transforms;
};
[shader("vertex")]
VertexOutput main_vs(
    VertexInput input,
    uint instance : SV_InstanceID,
    uniform PC pc
) {
    VertexOutput o = {};

    float4x4 model = pc.transforms[instance];
    float4 world_pos = mul(model, float4(input.position, 1.0));

    o.position  = mul(pc.frame_ubo->view_projection, world_pos);
    o.world_pos = world_pos.xyz;

    // Normal
    float3 local_normal = normalize(unpackUnorm3x10_1x2(input.normal).xyz);
    o.normal = normalize(mul((float3x3)model, local_normal));

    // UVs
    UV4 uvpack = unpack_uv_8_8_8_8(input.uvs);

    // Reconstruct quad UV from position (example)
    float2 uv_lerp = saturate(input.position.xy * 0.5 + 0.5);
    o.uv = lerp(uvpack.uv0, uvpack.uv1, uv_lerp);

    o.color = float3(0.1, 0.9, 0.1);

    return o;
}

// -----------------------------------------------------------------------------
// Fragment Shader
// -----------------------------------------------------------------------------
[shader("fragment")]
float4 main_fs(VertexOutput input, uniform PC pc) : SV_Target0
{
    // Inputs
    float3 albedo = input.color;
    float3 N = normalize(input.normal);

    // Sun (world space)
    float3 sun_dir = normalize(pc.frame_ubo->sun_dir);
    float  sun_intensity = pc.frame_ubo->sun_intensity;

    // Light comes *from* the sun direction
    float3 L = -sun_dir;

    // View direction
    float3 V = normalize(
        pc.frame_ubo->camera_position.xyz - input.world_pos
    );

    // Lambert
    float NdotL = saturate(dot(N, L));

    // Cheap Blinn-Phong spec
    float3 H = normalize(L + V);
    float spec = pow(saturate(dot(N, H)), 32.0);

    // Ambient + direct
    float3 lighting =
        albedo * (0.15 + NdotL * sun_intensity) +
        spec * sun_intensity;

    return float4(lighting, 1.0);
}
