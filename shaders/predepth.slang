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
    float4x4* transforms;
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
    float4x4 model = pc.transforms[instance_index];
    float4 world_pos = mul(model, float4(input.position, 1.0));
    o.position  = mul(pc.frame_ubo->view_projection, world_pos);
    return o;
}


[shader("vertex")]
float4 main_vs(VertexInput input,  uint instance: SV_InstanceID, uniform PC pc): SV_Position {
    VertexOutput output = {};
    return mul(pc.frame_ubo->view_projection, mul(pc.transforms[instance] ,float4(input.position, 1.0)));
}

[shader("fragment")]
void main_fs() {
    
}
