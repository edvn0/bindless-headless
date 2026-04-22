import pointer;
import common;

static const float  k_grid_size                    = 100.0;
static const float  k_grid_cell_size               = 0.025;
static const float  k_grid_min_pixels_between_cells = 2.0;
static const float4 k_grid_color_thin              = float4(0.5, 0.5, 0.5, 1.0);
static const float4 k_grid_color_thick             = float4(0.0, 0.0, 0.0, 1.0);

struct PushConstants {
    DeviceReadPtr<UBO> frame_ubo;
    float4 origin;
};

struct VertexOutput {
    float4 sv_pos  : SV_Position;
    [[vk::location(0)]] float2 uv;
    [[vk::location(1)]] float2 cam_pos;
};

static const float3 k_quad_positions[4] = {
    float3(-1.0, 0.0, -1.0),
    float3( 1.0, 0.0, -1.0),
    float3( 1.0, 0.0,  1.0),
    float3(-1.0, 0.0,  1.0),
};

static const uint k_quad_indices[6] = { 0, 1, 2, 2, 3, 0 };

[shader("vertex")]
VertexOutput vs_main(uint vertex_id : SV_VertexID, uniform PushConstants pc) {
    float3 position  = k_quad_positions[k_quad_indices[vertex_id]] * k_grid_size;
    position.x      += pc.frame_ubo->camera_position.x;
    position.z      += pc.frame_ubo->camera_position.z;
    position        += pc.origin.xyz;

    VertexOutput o;
    o.sv_pos  = mul(pc.frame_ubo->view_projection, float4(position, 1.0));
    o.uv      = position.xz;
    o.cam_pos = pc.frame_ubo->camera_position.xz;

    o.sv_pos.z += 0.00001 * o.sv_pos.w;

    return o;
}

float max2(float2 v) { return max(v.x, v.y); }

float4 grid_color(float2 uv, float2 cam_pos) {
    float2 dudv = float2(
        length(float2(ddx(uv.x), ddy(uv.x))),
        length(float2(ddx(uv.y), ddy(uv.y)))
    );

    float lod_level = max(0.0, log10(length(dudv) * k_grid_min_pixels_between_cells / k_grid_cell_size) + 1.0);
    float lod_fade  = frac(lod_level);

    float lod0 = k_grid_cell_size * pow(10.0, floor(lod_level));
    float lod1 = lod0 * 10.0;
    float lod2 = lod1 * 10.0;

    dudv *= 4.0;
    uv   += dudv * 0.5;

    float lod0a = max2(1.0 - abs(saturate((frac(uv / lod0) * lod0) / dudv) * 2.0 - 1.0));
    float lod1a = max2(1.0 - abs(saturate((frac(uv / lod1) * lod1) / dudv) * 2.0 - 1.0));
    float lod2a = max2(1.0 - abs(saturate((frac(uv / lod2) * lod2) / dudv) * 2.0 - 1.0));

    uv -= cam_pos;

    float4 c = lod2a > 0.0 ? k_grid_color_thick
             : lod1a > 0.0 ? lerp(k_grid_color_thick, k_grid_color_thin, lod_fade)
             :                k_grid_color_thin;

    float opacity_falloff  = 1.0 - saturate(length(uv) / k_grid_size);
    c.a *= (lod2a > 0.0 ? lod2a
          : lod1a > 0.0 ? lod1a
          :                lod0a * (1.0 - lod_fade)) * opacity_falloff;

    return c;
}

[shader("fragment")]
float4 fs_main(VertexOutput i) : SV_Target {
    return grid_color(i.uv, i.cam_pos);
}
