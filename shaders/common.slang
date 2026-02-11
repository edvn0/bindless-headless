struct PointLight
{
    float4 position_radius;
    float4 colour_intensity;
};

struct LightVisibility
{
    uint x0, x1;
    uint y0, y1;
    uint z0, z1;
    uint is_visible; 
    uint _pad;
};

struct UBO {
    float4x4 view;
    float4x4 projection;
    float4x4 view_projection;
    float4x4 inv_projection;
    float4x4 inv_view_projection;
    float4 camera_position;
    float4 frustum_planes[6];
    float4 sun_direction_intensity;
    float2 viewport_size;

    property sun_dir : float3
    {
        get { return sun_direction_intensity.xyz; }
    }

    property sun_intensity : float
        {
            get { return sun_direction_intensity.w; }
        }
};

struct Transform {
    float3x4 m;
};

/*float2 unpack_uvs(uint packed_uvs)
{
    uint x_bits = packed_uvs & 0xFFFF;
    uint y_bits = (packed_uvs >> 16) & 0xFFFF;
    
    float x = float(x_bits) / 65535.0;
    float y = float(y_bits) / 65535.0;
    
    return float2(x, y);
}*/

float2 unpack_uvs(uint packed_uvs)
{
    // Native Slang/HLSL function for Unorm 2x16
    printf("%u", packed_uvs);
    return unpackUnorm2x16ToFloat(packed_uvs).yx;
}

float4 unpackSnorm3x10_1x2(uint packed)
{
    // Extract 10-bit fields
    int x = int(packed & 0x3FF);           // bits 0..9
    int y = int((packed >> 10) & 0x3FF);   // bits 10..19
    int z = int((packed >> 20) & 0x3FF);   // bits 20..29
    int w = int((packed >> 30) & 0x3);     // bits 30..31

    // Sign-extend 10-bit values
    if (x >= 512) x -= 1024;
    if (y >= 512) y -= 1024;
    if (z >= 512) z -= 1024;

    // Convert to normalized floats
    return float4(
        float(x) / 511.0,
        float(y) / 511.0,
        float(z) / 511.0,
        float(w) / 3.0
    );
}


struct IndirectCommand {
    uint index_count;
    uint instance_count;
    uint first_index;
    int vertex_offset;
    uint first_instance;
};

struct IndirectMeshCommand {
    uint wgx;
    uint wgy;
    uint wgz;
};

struct Cube {
    float4 pos_radius;
    float4 colour_intensity;
};

static const uint FLAG_ALBEDO_MAP    = 1 << 0;
static const uint FLAG_NORMAL_MAP    = 1 << 1;
static const uint FLAG_ROUGHNESS_MAP = 1 << 2;
static const uint FLAG_METALLIC_MAP  = 1 << 3;
static const uint FLAG_OCCLUSION_MAP = 1 << 4;
static const uint FLAG_EMISSIVE_MAP  = 1 << 5;
static const uint FLAG_IS_ALPHA_TESTED  = 1 << 6;

struct Material {
    uint albedo_map;
    float4 albedo_factor;
    uint normal_map;
    uint roughness_map;
    float roughness_factor;
    uint metallic_map;
    float metallic_factor;
    uint occlusion_map;
    uint emissive_map;
    float3 emissive_factor;
    uint flags;

    bool has_albedo_map()  { return (flags & FLAG_ALBEDO_MAP) != 0; }
    bool has_normal_map()  { return (flags & FLAG_NORMAL_MAP) != 0; }
    bool has_roughness_map()  { return (flags & FLAG_ROUGHNESS_MAP) != 0; }
    bool has_metallic_map()  { return (flags & FLAG_METALLIC_MAP) != 0; }
    bool has_occlusion_map()  { return (flags & FLAG_OCCLUSION_MAP) != 0; }
    bool has_emissive_map()  { return (flags & FLAG_EMISSIVE_MAP) != 0; }
    bool is_alpha_tested()  { return (flags & FLAG_IS_ALPHA_TESTED) != 0; }
};


[[vk::binding(0, 0)]]
Texture2D<float4> sampled_images[];

[[vk::binding(1, 0)]]
SamplerState samplers[];

[[vk::binding(2, 0)]]
RWTexture2D<float4> storage_images[];

[[vk::binding(3, 0)]] 
SamplerComparisonState comparison_samplers[];

float4 sample_texture_lod(float2 uv, uint image_index, uint sampler_index = 0, float lod)
{
    uint tex_idx = NonUniformResourceIndex(image_index);
    uint samp_idx = NonUniformResourceIndex(sampler_index);

    return sampled_images[tex_idx].SampleLevel(samplers[samp_idx], uv, lod);
}

float4 sample_texture(float2 uv, uint image_index, uint sampler_index = 0)
{
    uint tex_idx = NonUniformResourceIndex(image_index);
    uint samp_idx = NonUniformResourceIndex(sampler_index);

    return sampled_images[tex_idx].Sample(samplers[samp_idx], uv);
}
