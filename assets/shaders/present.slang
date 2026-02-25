import common;

struct PresentPushConstants
{
    uint image_index;     // tonemapped UNORM texture index
    uint sampler_index;   // sampler index
    uint dst_is_srgb;     // 1 if swapchain is *_SRGB, else 0
};

struct VSOut
{
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
};

struct PSOut { float4 color : SV_Target0; };

float3 linear_to_srgb(float3 x)
{
    x = max(x, 0.0);
    float3 lo = x * 12.92;
    float3 hi = 1.055 * pow(x, 1.0 / 2.4) - 0.055;
    return select(x <= 0.0031308, lo, hi);
}

[shader("fragment")]
void present_fs(VSOut i, uniform PresentPushConstants pc, out PSOut o)
{
    uint idx      = NonUniformResourceIndex(pc.image_index);
    uint samp_idx = NonUniformResourceIndex(pc.sampler_index);

    float4 c = sampled_images[idx].Sample(samplers[samp_idx], i.uv);

   /* if (pc.dst_is_srgb == 0)
    {
        c.rgb = linear_to_srgb(c.rgb);
    }*/

    o.color = c;
}
