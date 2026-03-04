import common;

// ── Helpers ───────────────────────────────────────────────────────────────────

float luminance(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

float threshold_weight(float lum, float threshold, float knee) {
    float lower = threshold - knee;
    float upper = threshold + knee;
    if (lum < lower) return 0.0;
    if (lum > upper) return 1.0;
    float t = (lum - lower) / (2.0 * knee);
    return t * t * (3.0 - 2.0 * t);
}

// ── 1. Threshold ──────────────────────────────────────────────────────────────

struct ThresholdPC {
    uint  src_index;
    uint  dst_index;
    uint  sampler_index;
    float threshold;
    float knee;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void bloom_threshold_cs(
    uint3                              dispatch_id : SV_DispatchThreadID,
    uniform ThresholdPC pc)
{
    RWTexture2D<float4> dst = get_storage_image(pc.dst_index);

    uint2 dims;
    dst.GetDimensions(dims.x, dims.y);
    if (any(dispatch_id.xy >= dims)) return;

    float2 uv     = (float2(dispatch_id.xy) + 0.5) / float2(dims);
    float4 colour = sample_texture_lod(uv, pc.src_index, 0.0, pc.sampler_index);
    float  weight = threshold_weight(luminance(colour.rgb), pc.threshold, pc.knee);

    dst[dispatch_id.xy] = float4(colour.rgb * weight, 1.0);
}

// ── 2. Downsample (Jimenez 13-tap) ───────────────────────────────────────────

struct DownsamplePC {
    uint   src_index;
    uint   dst_index;
    uint   sampler_index;
    float2 src_texel_size;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void bloom_downsample_cs(
    uint3                                dispatch_id : SV_DispatchThreadID,
    uniform DownsamplePC   pc)
{
    RWTexture2D<float4> dst = get_storage_image(pc.dst_index);

    uint2 dims;
    dst.GetDimensions(dims.x, dims.y);
    if (any(dispatch_id.xy >= dims)) return;

    float2 uv = (float2(dispatch_id.xy) + 0.5) / float2(dims);
    float2 e  = pc.src_texel_size;

    // Corner taps (weight 0.125 each, ×4 = 0.5)
    float4 a = sample_texture_lod(uv + float2(-2, -2) * e, pc.src_index, 0.0, pc.sampler_index);
    float4 c = sample_texture_lod(uv + float2( 2, -2) * e, pc.src_index, 0.0, pc.sampler_index);
    float4 h = sample_texture_lod(uv + float2(-2,  2) * e, pc.src_index, 0.0, pc.sampler_index);
    float4 j = sample_texture_lod(uv + float2( 2,  2) * e, pc.src_index, 0.0, pc.sampler_index);

    // Cross taps (weight 0.125 each, ×4 = 0.5)
    float4 b = sample_texture_lod(uv + float2( 0, -2) * e, pc.src_index, 0.0, pc.sampler_index);
    float4 d = sample_texture_lod(uv + float2(-2,  0) * e, pc.src_index, 0.0, pc.sampler_index);
    float4 g = sample_texture_lod(uv + float2( 2,  0) * e, pc.src_index, 0.0, pc.sampler_index);
    float4 i = sample_texture_lod(uv + float2( 0,  2) * e, pc.src_index, 0.0, pc.sampler_index);

    // Inner bilinear quads (weight 0.5 each, ×4 = 2.0)
    float4 k = sample_texture_lod(uv + float2(-1, -1) * e, pc.src_index, 0.0, pc.sampler_index);
    float4 l = sample_texture_lod(uv + float2( 1, -1) * e, pc.src_index, 0.0, pc.sampler_index);
    float4 m = sample_texture_lod(uv + float2(-1,  1) * e, pc.src_index, 0.0, pc.sampler_index);
    float4 n = sample_texture_lod(uv + float2( 1,  1) * e, pc.src_index, 0.0, pc.sampler_index);

    // Total weight = 0.5 + 0.5 + 2.0 = 3.0, so normalise by 1/3
    float4 result =
          (0.125 / 3.0) * (a + c + h + j)
        + (0.125 / 3.0) * (b + d + g + i)
        + (0.5   / 3.0) * (k + l + m + n);

    dst[dispatch_id.xy] = result;
}

// ── 3. Upsample (3×3 tent) ───────────────────────────────────────────────────

struct UpsamplePC {
    uint  src_index;
    uint  accumulate_index;
    uint  sampler_index;
    float filter_radius;
    float strength;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void bloom_upsample_cs(
    uint3                              dispatch_id : SV_DispatchThreadID,
    uniform UpsamplePC   pc)
{
    RWTexture2D<float4> dst = get_storage_image(pc.accumulate_index);

    uint2 dims;
    dst.GetDimensions(dims.x, dims.y);
    if (any(dispatch_id.xy >= dims)) return;

    float2 uv = (float2(dispatch_id.xy) + 0.5) / float2(dims);

    // Scale radius by texel size so offsets are uniform in pixel space
    float2 r = pc.filter_radius * float2(1.0 / float(dims.x), 1.0 / float(dims.y));

    float4 result =
          (1.0 / 16.0) * sample_texture_lod(uv + float2(-r.x, -r.y), pc.src_index, 0.0, pc.sampler_index)
        + (2.0 / 16.0) * sample_texture_lod(uv + float2( 0,   -r.y), pc.src_index, 0.0, pc.sampler_index)
        + (1.0 / 16.0) * sample_texture_lod(uv + float2( r.x, -r.y), pc.src_index, 0.0, pc.sampler_index)
        + (2.0 / 16.0) * sample_texture_lod(uv + float2(-r.x,  0  ), pc.src_index, 0.0, pc.sampler_index)
        + (4.0 / 16.0) * sample_texture_lod(uv + float2( 0,    0  ), pc.src_index, 0.0, pc.sampler_index)
        + (2.0 / 16.0) * sample_texture_lod(uv + float2( r.x,  0  ), pc.src_index, 0.0, pc.sampler_index)
        + (1.0 / 16.0) * sample_texture_lod(uv + float2(-r.x,  r.y), pc.src_index, 0.0, pc.sampler_index)
        + (2.0 / 16.0) * sample_texture_lod(uv + float2( 0,    r.y), pc.src_index, 0.0, pc.sampler_index)
        + (1.0 / 16.0) * sample_texture_lod(uv + float2( r.x,  r.y), pc.src_index, 0.0, pc.sampler_index);

    dst[dispatch_id.xy] = dst[dispatch_id.xy] + result * pc.strength;
}
