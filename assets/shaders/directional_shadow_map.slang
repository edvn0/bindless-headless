import common;
import maths;

struct ShadowPC {
    float4x4   light_view_proj;
    Transform* transforms;
    uint*      draw_material_ids;
    Material*  materials;
    uint       base_draw_id;
    uint       sampler_index;
};

struct VertexInput {
    float3 position;
};

struct VSOut {
    float4 position : SV_Position;
    nointerpolation uint material_id : TEXCOORD0;
};

[shader("vertex")]
VSOut shadow_vs_mdi(
    VertexInput input,
    uint draw_id      : SV_DrawIndex,
    uint instance_id  : SV_InstanceID,
    uint base_instance: SV_StartInstanceLocation,
    uniform ShadowPC pc)
{
    VSOut o = {};
    
    uint global_draw = pc.base_draw_id + draw_id;
    o.material_id = pc.draw_material_ids[global_draw];
    uint instance_index = instance_id + base_instance;
    
    float3x4 model_3x4 = pc.transforms[instance_index].m;
    float4x4 model = float4x4(
        model_3x4._m00_m01_m02_m03,
        model_3x4._m10_m11_m12_m13,
        model_3x4._m20_m21_m22_m23,
        float4(0, 0, 0, 1)
    );
    
    float4 world_pos = mul(model, float4(input.position, 1.0));
    o.position = mul(pc.light_view_proj, world_pos);
    
    return o;
}

[shader("fragment")]
void shadow_fs_main(VSOut i, uniform ShadowPC pc)
{
    Material mat = pc.materials[i.material_id];
    
    if (mat.has_albedo_map()) {
        float alpha = 1.0f- mat.albedo_factor.a;
        
        if (alpha < 0.5f) {
            discard;
        }
    }
    
}