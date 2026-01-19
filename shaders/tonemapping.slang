import common;

struct TonemapPushConstants
{
    uint image_index;   // HDR texture index
    uint sampler_index; // sampler index
};

struct VSOut
{
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
};

[shader("vertex")]
VSOut vs_main(uint vertex_id : SV_VertexID)
{
    // Fullscreen triangle (no vertex buffer)
    float2 pos = float2(
        (vertex_id == 2) ?  3.0 : -1.0,
        (vertex_id == 1) ?  3.0 : -1.0
    );

    VSOut o;
    o.position = float4(pos, 0.0, 1.0);
    o.uv = pos * 0.5 + 0.5;
    return o;
}

struct PSOut
{
    float4 color : SV_Target0;
};

[shader("fragment")]
PSOut ps_main(VSOut i, uniform TonemapPushConstants pc)
{
    uint idx = NonUniformResourceIndex(pc.image_index);
    uint samp_idx = NonUniformResourceIndex(pc.sampler_index);

    float4 hdr = sampled_images[idx].SampleLevel(samplers[samp_idx], i.uv, 0);

    // Reinhard tonemap: HDR -> [0,1] linear
    float3 ldr_rgb = hdr.rgb / (hdr.rgb + 1.0);

    PSOut o;
    o.color = float4(ldr_rgb, hdr.a); // linear RGB, no gamma
    return o;
}
