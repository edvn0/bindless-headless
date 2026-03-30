import common;
import maths;
import constants;
import pointer;

struct PC {
    UBO* frame_ubo;
    DeviceReadPtr<InstanceData> instance_data;
    uint* draw_material_ids;
    Material* materials;
    uint       base_draw_id;
    uint       sampler_index;
};

struct VertexInput {
    float3 position;
    uint uv0;
    uint uv1;
};

struct VSOut {
    float4 position : SV_Position;
    nointerpolation uint material_id : TEXCOORD0;
    float4 uvs : TEXCOORD1;
};

[shader("vertex")]
VSOut main_vs_mdi(
    VertexInput input,
    uint draw_id      : SV_DrawIndex,
    uint instance_id  : SV_InstanceID,
    uint base_instance: SV_StartInstanceLocation,
    uniform PC pc)
{
    VSOut o = {};

    // 1. Resolve Identifiers
    uint global_draw = pc.base_draw_id + draw_id;
    o.material_id = pc.draw_material_ids[global_draw];
    uint instance_index = instance_id + base_instance;

    // 2. Transform Position
    float3x4 model_3x4 = pc.instance_data[instance_index].transform.m;
    float4x4 model = float4x4(
        model_3x4._m00_m01_m02_m03,
        model_3x4._m10_m11_m12_m13,
        model_3x4._m20_m21_m22_m23,
        float4(0, 0, 0, 1)
    );

    float4 world_pos = mul(model, float4(input.position, 1.0));
    o.position = mul(pc.frame_ubo->view_projection, world_pos);
    o.uvs = float4(unpackHalf2x16ToFloat(input.uv0), unpackHalf2x16ToFloat(input.uv1));

    return o;
}

[shader("fragment")]
void fs_main(VSOut i, uniform PC pc)
{
    Material mat = pc.materials[i.material_id];
    float alpha = sample_texture(i.uvs.xy, mat.albedo_map, 0).a;

    if (alpha < alpha_cutoff) {
        discard;
    }
}