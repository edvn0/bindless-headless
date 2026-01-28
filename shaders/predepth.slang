import common;
import maths;

struct VertexOutput {
    float4 position;
};

struct VertexInput {
    float3 position;
};


struct PC {
    UBO* frame_ubo;
    Transform* transforms;
    uint* draw_material_ids;
    uint  base_draw_id;
};
[shader("vertex")]
VertexOutput main_vs_mdi(
    VertexInput input,
    uint draw_id      : SV_DrawIndex,
    uint instance_id  : SV_InstanceID,
    uint base_instance: SV_StartInstanceLocation,
    uniform PC pc)
{
    VertexOutput o = {};
    uint instance_index = instance_id + base_instance;
    float3x4 model_3x4 = pc.transforms[instance_index].m;
       float4x4 model = float4x4(
           model_3x4._m00_m01_m02_m03,
           model_3x4._m10_m11_m12_m13,
           model_3x4._m20_m21_m22_m23,
           float4(0, 0, 0, 1)
       );    float4 world_pos = mul(model, float4(input.position, 1.0));
    o.position  = mul(pc.frame_ubo->view_projection, world_pos);
    return o;
}


[shader("vertex")]
float4 main_vs(VertexInput input,  uint instance: SV_InstanceID, uniform PC pc): SV_Position {
    VertexOutput output = {};
    float3x4 model_3x4 = pc.transforms[instance].m;
       float4x4 model = float4x4(
           model_3x4._m00_m01_m02_m03,
           model_3x4._m10_m11_m12_m13,
           model_3x4._m20_m21_m22_m23,
           float4(0, 0, 0, 1)
       );    float4 world_pos = mul(model, float4(input.position, 1.0));
    return mul(pc.frame_ubo->view_projection,world_pos );
}

[shader("fragment")]
void main_fs() {

}
