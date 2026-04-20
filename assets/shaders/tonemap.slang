import common;

struct TonemapPushConstants {
    float exposure;
    uint  image_index;
    uint  sampler_index;
    uint  bloom_index;
    float bloom_strength;
};

struct VSOut {
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
};

[shader("vertex")]
VSOut vs_main(uint vertex_id : SV_VertexID) {
    float2 pos = float2(
        (vertex_id == 2) ?  3.0 : -1.0,
        (vertex_id == 1) ?  3.0 : -1.0
    );
    VSOut o;
    o.position = float4(pos, 0.0, 1.0);
    o.uv       = pos * 0.5 + 0.5;
    o.uv.y     = 1.0 - o.uv.y;
    return o;
}

struct PSOut {
    float4 color : SV_Target0;
};

[shader("fragment")]
void fs_main(VSOut i, uniform TonemapPushConstants pc, out PSOut o) {
    float3 hdr   = sample_texture_lod(i.uv, pc.image_index, 0.0, pc.sampler_index).rgb;
    float3 bloom = sample_texture_lod(i.uv, pc.bloom_index,  0.0, pc.sampler_index).rgb;

    float3 combined = hdr * pc.exposure + bloom * 0.01*pc.bloom_strength;

    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;

    float L   = dot(combined, float3(0.2126, 0.7152, 0.0722));
    float L_t = (L * (a * L + b)) / (L * (c * L + d) + e);

    float3 ldr = saturate(combined * (L_t / max(L, 1e-5)));
    o.color = float4(ldr, 1.0);
}
