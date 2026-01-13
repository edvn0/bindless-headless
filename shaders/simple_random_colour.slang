[[vk::binding(0, 0)]]
Texture2D<float> sampled_images[];

[[vk::binding(1, 0)]]
SamplerState samplers[];

[[vk::binding(2, 0)]]
RWTexture2D<float4> storage_images[];

struct PushConstants {
    uint image_index;
    uint noise_index;
    uint noise_sampler_index;
};

float sample_noise(uint noise_idx, uint noise_sampler_index, uint2 pixel)
{
    uint w, h;
    sampled_images[noise_idx].GetDimensions(w, h);
    float2 uv = float2(pixel) / float2(w, h);
    return sampled_images[noise_idx].SampleLevel(samplers[noise_sampler_index], uv, 0.0);
}

uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

float random_float(uint seed) {
    return float(hash(seed)) / float(0xffffffffU);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uniform PushConstants pc, uint3 thread_id : SV_DispatchThreadID)
{
    uint idx = NonUniformResourceIndex(pc.image_index);
    uint noise_idx = NonUniformResourceIndex(pc.noise_index);
    uint noise_sampler_index = NonUniformResourceIndex(pc.noise_sampler_index);

    uint2 pixel = thread_id.xy;

    uint w, h;
    storage_images[idx].GetDimensions(w, h);

    if (pixel.x >= w || pixel.y >= h)
        return;

    float n = sample_noise(noise_idx, noise_sampler_index, pixel);

    // mix noise with random hash
    uint seed = pixel.y * w + pixel.x;
    float r = lerp(n, random_float(seed), 0.25);
    float g = lerp(n, random_float(seed + 1), 0.25);
    float b = n;

    storage_images[idx][pixel] = float4(r, g, b, 1.0);
}
