import common;
import pointer;

struct SkyboxPushConstants
{
    DeviceReadPtr<UBO> frame_ubo;
    uint cubemap_index;
    uint sampler_index;
}

struct VSOut
{
    float4 position : SV_Position;
    float3 direction : TEXCOORD0;
}

struct PSOut { float4 color : SV_Target0; }

[shader("vertex")]
VSOut cubemap_vs(uint vertex_id : SV_VertexID, uniform SkyboxPushConstants pc)
{
    float2 uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    float4 clip_pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);

    var ubo = *pc.frame_ubo;

    float4 world = mul(ubo.inv_view_projection_no_translation, clip_pos);

    VSOut o;
    o.position  = float4(clip_pos.xy, 0.0, 1.0);
    o.direction = world.xyz;
    return o;
}

[shader("fragment")]
void cubemap_fs(VSOut i, uniform SkyboxPushConstants pc, out PSOut o)
{
    uint cube_idx = NonUniformResourceIndex(pc.cubemap_index);
    uint samp_idx = NonUniformResourceIndex(pc.sampler_index);

    float3 dir = normalize(i.direction);
    var image = sampled_cubemaps[cube_idx];
    o.color = image.Sample(samplers[samp_idx], dir);
}