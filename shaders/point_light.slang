import common;
import maths;

struct VertexOutput {
    float4 position  : SV_Position;
    float3 world_pos : TEXCOORD0;
    float3 normal    : TEXCOORD1;
    float2 uv        : TEXCOORD2;
   nointerpolation uint   material  : TEXCOORD3; // pass along if you want
};

struct VertexInput {
    float3 position;
    uint normal;
    uint uvs;
};

struct PC {
    UBO* frame_ubo;
    Transform* transforms;
    uint* draw_material_ids;
    Material* materials;
    uint  base_draw_id;
};
[shader("vertex")]
VertexOutput main_vs(
    VertexInput input,
    uint instance : SV_InstanceID,
    uniform PC pc
) {
    VertexOutput o = {};

    float3x4 model_3x4 = pc.transforms[instance].m;
       float4x4 model = float4x4(
           model_3x4._m00_m01_m02_m03,
           model_3x4._m10_m11_m12_m13,
           model_3x4._m20_m21_m22_m23,
           float4(0, 0, 0, 1)
       );    float4 world_pos = mul(model, float4(input.position, 1.0));

    o.position  = mul(pc.frame_ubo->view_projection, world_pos);
    o.world_pos = world_pos.xyz;

    // Normal
    float3 local_normal = normalize(unpackSnorm3x10_1x2(input.normal).xyz);
    o.normal = normalize(mul((float3x3)model, local_normal));

    // UVs
    UV4 uvpack = unpack_uv_8_8_8_8(input.uvs);

    // Reconstruct quad UV from position (example)
    float2 uv_lerp = saturate(input.position.xy * 0.5 + 0.5);
    o.uv = lerp(uvpack.uv0, uvpack.uv1, uv_lerp);

    return o;
}

[shader("vertex")]
VertexOutput main_vs_mdi(
    VertexInput input,
    uint draw_id      : SV_DrawIndex,
    uint instance_id  : SV_InstanceID,
    uint base_instance: SV_StartInstanceLocation,
    uniform PC pc)
{
    VertexOutput o = {};

    uint global_draw = pc.base_draw_id + draw_id;
    uint material_id = pc.draw_material_ids[global_draw];

    uint instance_index = instance_id + base_instance;

    float3x4 model_3x4 = pc.transforms[instance_index].m;
       float4x4 model = float4x4(
           model_3x4._m00_m01_m02_m03,
           model_3x4._m10_m11_m12_m13,
           model_3x4._m20_m21_m22_m23,
           float4(0, 0, 0, 1)
       );    float4 world_pos = mul(model, float4(input.position, 1.0));


    o.position  = mul(pc.frame_ubo->view_projection, world_pos);
    o.world_pos = world_pos.xyz;

    float3 local_normal = normalize(unpackSnorm3x10_1x2(input.normal).xyz);
    o.normal = normalize(mul((float3x3)model, local_normal));

    UV4 uvpack = unpack_uv_8_8_8_8(input.uvs);
    float2 uv_lerp = saturate(input.position.xy * 0.5 + 0.5);
    o.uv = lerp(uvpack.uv0, uvpack.uv1, uv_lerp);

    o.material = material_id;
    return o;
}

// -----------------------------------------------------------------------------
// Fragment Shader
// -----------------------------------------------------------------------------
[shader("fragment")]
float4 main_fs(VertexOutput input, uniform PC pc) : SV_Target0
{
    // Inputs
    float3 albedo = {0.0F, 1.0F, 0.0F};
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
