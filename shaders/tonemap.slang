import common;

struct TonemapPushConstants
{
    float exposure;
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
    o.uv.y = 1 - o.uv.y;
    return o;
}

struct PSOut
{
    float4 color : SV_Target0;
};

[shader("fragment")]
PSOut fs_main(VSOut i, uniform TonemapPushConstants pc)
{
    uint idx = NonUniformResourceIndex(pc.image_index);
    uint samp_idx = NonUniformResourceIndex(pc.sampler_index);
    float4 hdr = sampled_images[idx].Sample(samplers[samp_idx], i.uv);
    
    // Apply exposure first
    float3 color = hdr.rgb * pc.exposure;
    
    // ACES filmic tonemap (better than Reinhard, preserves colors)
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    float3 ldr_rgb = saturate((color * (a * color + b)) / (color * (c * color + d) + e));
    
    PSOut o;
    o.color =float4(hdr.xyz,1); // output linear, hardware converts to sRGB
    return o;
}
