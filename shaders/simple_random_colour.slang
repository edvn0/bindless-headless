[[vk::binding(2, 0)]]
RWTexture2D<float4> storage_images[];

struct PushConstants {
  uint image_index;
  uint padding;
};

uint hash(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

float random_float(uint seed) { return float(hash(seed)) / float(0xffffffffU); }

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uniform PushConstants pc, uint3 thread_id: SV_DispatchThreadID) {
  uint idx = NonUniformResourceIndex(pc.image_index);

  uint2 pixel = thread_id.xy;

  uint w, h;
  storage_images[idx].GetDimensions(w, h);

  if (pixel.x >= w || pixel.y >= h)
    return;

  uint seed = pixel.y * w + pixel.x;

  float r = random_float(seed);
  float g = random_float(seed + 1);
  float b = random_float(seed + 2);

  storage_images[idx][pixel] = float4(r, g, b, 1.0);
}
