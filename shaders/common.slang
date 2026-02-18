struct PointLight
{
    float4 position_radius;
    float4 colour_intensity;

    property radius : float {
        get { return position_radius.w; }
    }
    property position : float3 {
        get { return position_radius.xyz; }
    }
    property intensity : float {
        get { return colour_intensity.w; }
    }
    property colour : float3 {
        get { return colour_intensity.xyz; }
    }
};

struct LightVisibility
{
    uint x0, x1;
    uint y0, y1;
    uint z0, z1;
    uint is_visible; 
    uint _pad;
};

struct Cluster
{
    uint light_offset;
    uint light_count;
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

float4 sample_texture_lod(float2 uv, uint image_index,  float lod, uint sampler_index = 0)
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

RWTexture2D<float4> get_storage_image(uint image_index) {
    return storage_images[NonUniformResourceIndex(image_index)];
}

float3 reconstruct_world_position_from_depth(float2 uv, float depth, Ptr<UBO, Access.Read, AddressSpace.Device> ubo)
{
    float2 ndc_xy = uv * 2.0 - 1.0;

    ndc_xy.y = -ndc_xy.y; 

    float4 clip = float4(ndc_xy, depth, 1.0);
    float4 world = mul(ubo->inv_view_projection, clip);
    
    return world.xyz / world.w;
}

float calculate_shadow(float3 world_pos, float4x4 shadow_matrix, uint tex_idx, uint comp_samp_idx)
{
    float4 shadow_coord = mul(shadow_matrix, float4(world_pos, 1.0));
    float3 ndc = shadow_coord.xyz / shadow_coord.w;
    
    float2 shadow_uv = float2(
        ndc.x * 0.5 + 0.5,
        ndc.y * -0.5 + 0.5  
    );
    
    if (any(shadow_uv < 0.0) || any(shadow_uv > 1.0))
        return 1.0;
    
    uint t_idx = NonUniformResourceIndex(tex_idx);
    uint cs_idx = NonUniformResourceIndex(comp_samp_idx);
    
    const float bias = 0.0005; 
    float reference_depth = ndc.z - bias;      
    reference_depth = saturate(reference_depth);
    float lit_factor = sampled_images[t_idx].SampleCmp(
        comparison_samplers[cs_idx],
        shadow_uv,
        reference_depth
    );
    
    return lit_factor;
}