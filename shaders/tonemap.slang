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
    // Fullscreen triangle
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
    // --- Sample HDR ---
    uint idx = NonUniformResourceIndex(pc.image_index);
    uint samp_idx = NonUniformResourceIndex(pc.sampler_index);

    // NOTE: sampled_images MUST be Texture2D<float4>
    float4 hdr = sampled_images[idx].Sample(samplers[samp_idx], i.uv);

    // --- Apply exposure ---
    float3 color = hdr.rgb * pc.exposure;

    // --- ACES Filmic Tonemap (luminance-aware) ---
    // ACES constants
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;

    // Compute luminance
    float L = dot(color, float3(0.2126, 0.7152, 0.0722));

    // ACES tonemap on luminance
    float L_t = (L * (a * L + b)) / (L * (c * L + d) + e);

    // Apply tonemapped luminance to preserve chroma
    float3 ldr_rgb = color * (L_t / max(L, 1e-5));

    // Clamp to [0,1] for rgba8_srgb output
    ldr_rgb = saturate(ldr_rgb);

    PSOut o;
    o.color = float4(ldr_rgb, 1.0);
    return o;
}