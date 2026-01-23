struct PointLight
{
    float4 position_radius;
    float4 colour_intensity;
};

struct UBO {
    float4x4 view;
    float4x4 projection;
    float4x4 view_projection;
    float4x4 inv_projection;
    float4 camera_position;
    float4 frustum_planes[6];
    float4 sun_direction_intensity;

    property sun_dir : float3
    {
        get { return sun_direction_intensity.xyz; }
    }

    property sun_intensity : float
        {
            get { return sun_direction_intensity.w; }
        }
};

struct UV4
{
    float2 uv0;
    float2 uv1;
};

UV4 unpack_uv_8_8_8_8(uint packed)
{
    float u0 = float( packed        & 0xFF) / 255.0;
    float v0 = float((packed >>  8)  & 0xFF) / 255.0;
    float u1 = float((packed >> 16)  & 0xFF) / 255.0;
    float v1 = float((packed >> 24)  & 0xFF) / 255.0;

    UV4 r;
    r.uv0 = float2(u0, v0);
    r.uv1 = float2(u1, v1);
    return r;
}

float4 unpackUnorm3x10_1x2(uint packed)
{
    // Extract 10-bit fields
    uint x = packed & 0x3FF;           // bits 0..9
    uint y = (packed >> 10) & 0x3FF;   // bits 10..19
    uint z = (packed >> 20) & 0x3FF;   // bits 20..29
    uint w = (packed >> 30) & 0x3;     // bits 30..31

    // Convert to normalized floats
    return float4(
        float(x) / 1023.0,
        float(y) / 1023.0,
        float(z) / 1023.0,
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
}

[[vk::binding(0, 0)]]
Texture2D<float4> sampled_images[];

[[vk::binding(1, 0)]]
SamplerState samplers[];

[[vk::binding(2, 0)]]
RWTexture2D<float4> storage_images[];