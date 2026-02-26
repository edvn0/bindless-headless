import common;
import maths;
import pointer;
import constants;

[[vk::constant_id(0)]]
const int MAX_WAVES_PER_GROUP = 4;
[[vk::constant_id(1)]]
const int THREADS_PER_GROUP = 64;

static const uint SSAO_KERNEL_SIZE = 32;
static const uint SSAO_NOISE_SIZE  = 4;

static const float3 KERNEL[SSAO_KERNEL_SIZE] = {
    float3( 0.0490,  0.0556,  0.0290),
    float3(-0.0761,  0.0373,  0.0266),
    float3( 0.0340, -0.0774,  0.0543),
    float3(-0.0415,  0.0714,  0.0414),
    float3( 0.1042, -0.0331,  0.0821),
    float3( 0.0532,  0.1142,  0.0610),
    float3(-0.1287, -0.0523,  0.0391),
    float3( 0.0893, -0.1231,  0.0745),
    float3(-0.0227,  0.1456,  0.1012),
    float3( 0.1634,  0.0742,  0.0556),
    float3(-0.1521, -0.1043,  0.0824),
    float3( 0.0762, -0.1732,  0.1143),
    float3(-0.1943,  0.0823,  0.0932),
    float3( 0.1234,  0.1834,  0.1256),
    float3(-0.0934, -0.2012,  0.1432),
    float3( 0.2134, -0.0923,  0.1234),
    float3(-0.1823,  0.1534,  0.1634),
    float3( 0.0534,  0.2234,  0.1823),
    float3(-0.2312, -0.0634,  0.1534),
    float3( 0.1923, -0.1634,  0.1723),
    float3(-0.0723,  0.2412,  0.2012),
    float3( 0.2512,  0.0423,  0.1912),
    float3(-0.2112, -0.1423,  0.2134),
    float3( 0.1312, -0.2312,  0.2234),
    float3(-0.2712,  0.0912,  0.2023),
    float3( 0.0912,  0.2712,  0.2312),
    float3(-0.2912, -0.0712,  0.2112),
    float3( 0.2412, -0.2012,  0.2512),
    float3(-0.1512,  0.2912,  0.2712),
    float3( 0.3012,  0.1112,  0.2312),
    float3(-0.2612, -0.2112,  0.2912),
    float3( 0.1812, -0.3012,  0.3012),
};

struct PushConstants
{
    DeviceReadPtr<UBO> frame_ubo;
    uint noise_index;
    uint gbuffer0_index;
    uint gbuffer1_index;
    uint depth_index;
    uint ssao_output_index;
    uint sampler_index;
    float radius;
    float bias;
};

float3 reconstruct_view_pos(float2 uv, float depth, DeviceReadPtr<UBO> ubo)
{
    // Match reconstruct_world_position_from_depth: Y is flipped in NDC
    float2 ndc_xy = uv * 2.0 - 1.0;
    ndc_xy.y = -ndc_xy.y;

    float4 clip   = float4(ndc_xy, depth, 1.0);
    float4 view_h = mul(ubo->inv_projection, clip);
    return view_h.xyz / view_h.w;
}

[numthreads(8, 8, 1)]
[shader("compute")]
void ssao_compute(
    uint3 dispatch_id : SV_DispatchThreadID,
    uniform PushConstants pc)
{
    RWTexture2D<float4> ssao_out = get_storage_image(pc.ssao_output_index);

    uint2 dims;
    ssao_out.GetDimensions(dims.x, dims.y);

    if (dispatch_id.x >= dims.x || dispatch_id.y >= dims.y)
        return;

    float2 uv = (float2(dispatch_id.xy) + 0.5) / float2(dims);

    float  depth     = sample_texture_lod(uv, pc.depth_index, 0, pc.sampler_index).r;
    float3 view_pos  = reconstruct_view_pos(uv, depth, pc.frame_ubo);

    // gbuffer1.xy is oct-encoded world-space normal — decode then transform to view space
    float2 oct_u      = sample_texture_lod(uv, pc.gbuffer1_index, 0, pc.sampler_index).xy;
    float3 normal_ws  = oct_decode(oct_u);
    float3 normal_vs  = normalize(mul((float3x3)pc.frame_ubo->view, normal_ws));

    // Tile the 4x4 noise texture over the screen for per-pixel kernel rotation
    float2 noise_uv   = float2(dispatch_id.xy) / float(SSAO_NOISE_SIZE);
    float3 random_vec = sample_texture_lod(noise_uv, pc.noise_index, 0, pc.sampler_index).xyz * 2.0 - 1.0;

    // Gram-Schmidt orthonormalisation to build TBN aligned to the surface normal
    float3 tangent   = normalize(random_vec - normal_vs * dot(random_vec, normal_vs));
    float3 bitangent = cross(normal_vs, tangent);
    float3x3 tbn     = float3x3(tangent, bitangent, normal_vs);

    float occlusion = 0.0;

    for (uint i = 0; i < SSAO_KERNEL_SIZE; ++i)
    {
        float3 sample_vs  = mul(tbn, KERNEL[i]);
        sample_vs         = view_pos + sample_vs * pc.radius;

        // Project the view-space sample back to UV
        float4 offset  = mul(pc.frame_ubo->projection, float4(sample_vs, 1.0));
        offset.xyz    /= offset.w;

        // Projection gives the same Y-flipped NDC, undo it for UV
        float2 sample_uv = float2(offset.x, -offset.y) * 0.5 + 0.5;

        float  sample_depth  = sample_texture_lod(sample_uv, pc.depth_index, 0, pc.sampler_index).r;
        float3 sample_actual = reconstruct_view_pos(sample_uv, sample_depth, pc.frame_ubo);

        // Range check: suppress contribution from samples far from the shaded point
        float range_check = smoothstep(0.0, 1.0, pc.radius / abs(view_pos.z - sample_actual.z));

        // Reverse-Z: view_pos.z is negative (forward is -Z), larger magnitude = further away.
        // A sample occludes if the actual geometry at that UV is in front of (closer than)
        // the expected sample position — i.e. actual.z > sample_vs.z (less negative).
        occlusion += (sample_actual.z >= sample_vs.z + pc.bias ? 1.0 : 0.0) * range_check;
    }

    float ao = 1.0 - (occlusion / float(SSAO_KERNEL_SIZE));
    ssao_out[dispatch_id.xy] = float4(ao, ao, ao, 1.0);
}