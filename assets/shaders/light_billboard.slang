import pointer;
import common;

struct BillboardVertex {
    float4 position : SV_Position;
    float4 color : COLOR;
    float2 uv : TEXCOORD0;
};

struct BillboardPushConstants {
    DeviceReadPtr<UBO>        frame_ubo;
    DeviceReadPtr<PointLight> point_lights;
    uint                      light_count;
    uint                      texture_index;
    uint                      sampler_index;
    float                     world_size; // billboard size in world units e.g. 0.25
};

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void billboard_ms(
    uint tid : SV_GroupIndex,
    uniform BillboardPushConstants pc,
    out vertices BillboardVertex verts[128],
    out indices uint3 tris[64])
{
    const uint light_count = min(pc.light_count, 32u);
    SetMeshOutputCounts(light_count * 4, light_count * 2);

    if (tid >= light_count) return;

    PointLight light = pc.point_lights[tid];
    float3 light_pos = light.position_radius.xyz;
    float3 color     = light.colour_intensity.rgb;

    float3 right = float3(pc.frame_ubo->view._m00, pc.frame_ubo->view._m10, pc.frame_ubo->view._m20);
    float3 up    = float3(pc.frame_ubo->view._m01, pc.frame_ubo->view._m11, pc.frame_ubo->view._m21);

    float h = pc.world_size * 0.5;

    const uint base_vert = tid * 4;
    const uint base_tri  = tid * 2;

    verts[base_vert + 0].position = mul(pc.frame_ubo->view_projection, float4(light_pos + (-right - up) * h, 1.0));
    verts[base_vert + 1].position = mul(pc.frame_ubo->view_projection, float4(light_pos + ( right - up) * h, 1.0));
    verts[base_vert + 2].position = mul(pc.frame_ubo->view_projection, float4(light_pos + ( right + up) * h, 1.0));
    verts[base_vert + 3].position = mul(pc.frame_ubo->view_projection, float4(light_pos + (-right + up) * h, 1.0));

    verts[base_vert + 0].uv = float2(0, 1);
    verts[base_vert + 1].uv = float2(1, 1);
    verts[base_vert + 2].uv = float2(1, 0);
    verts[base_vert + 3].uv = float2(0, 0);

    // Tint verts by light colour so the icon takes on the light's colour
    [unroll] for (uint v = 0; v < 4; ++v)
        verts[base_vert + v].color = float4(color, 1.0);

    tris[base_tri + 0] = uint3(base_vert, base_vert + 1, base_vert + 2);
    tris[base_tri + 1] = uint3(base_vert, base_vert + 2, base_vert + 3);
}

[shader("fragment")]
float4 billboard_fs(BillboardVertex i, uniform BillboardPushConstants pc) : SV_Target0 {
    float4 tex = sample_texture(i.uv, pc.texture_index, pc.sampler_index);
    // Discard transparent pixels so depth is correct
    if (tex.a < 0.1) discard;
    return float4(tex.rgb * i.color.rgb, tex.a);
}