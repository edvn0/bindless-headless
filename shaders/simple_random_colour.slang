[[vk::binding(0, 0)]]
Texture2D<float4> textures[];

[[vk::binding(1, 0)]]
SamplerState samplers[];

[[vk::binding(2, 0)]]
RWTexture2D<float4> storage_images[];

struct PushConstants {
    uint image_index;
};

// Simple hash function for pseudo-random numbers
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
void main(uniform PushConstants* push, uint3 thread_id : SV_DispatchThreadID) {
    uint2 pixel_coord = thread_id.xy;
    
    // Get the image from the bindless array
    RWTexture2D<float4> output_image = storage_images[push->image_index];
    
    // Check bounds
    uint width, height;
    output_image.GetDimensions(width, height);
    
    if (pixel_coord.x >= width || pixel_coord.y >= height) {
        return;
    }
    
    // Generate unique seed per pixel
    uint seed = pixel_coord.y * width + pixel_coord.x;
    
    // Generate random RGB color
    float r = random_float(seed);
    float g = random_float(seed + 1);
    float b = random_float(seed + 2);
    
    output_image[pixel_coord] = float4(r, g, b, 1.0);
}