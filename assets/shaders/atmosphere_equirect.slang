import common;

struct AtmospherePushConstants
{
    float3 sun_direction;   // normalized, pointing toward sun
    float  sun_intensity;
    uint   output_cubemap_face; // 0-5, dispatch once per face
    uint   output_image_index;  // RWTexture2D index in storage_images
    uint   face_size;
}

// Atmosphere constants
static const float k_earth_radius     = 6360000.0;
static const float k_atmosphere_radius = 6420000.0;
static const float k_ray_origin_height = 100.0; // metres above surface

static const float3 k_beta_rayleigh = float3(5.5e-6, 13.0e-6, 22.4e-6); // per metre
static const float  k_beta_mie      = 21e-6;
static const float  k_h_rayleigh    = 8000.0; // scale height metres
static const float  k_h_mie         = 1200.0;
static const float  k_g             = 0.758;  // Mie asymmetry factor
static const int    k_view_samples  = 16;
static const int    k_light_samples = 8;

float3 face_uv_to_direction(uint face, float2 uv)
{
    // uv in [0,1] -> [-1,1]
    float2 ndc = uv * 2.0 - 1.0;
    float3 dir;
    switch (face)
    {
        case 0: dir = float3( 1.0,  -ndc.y, -ndc.x); break; // +X
        case 1: dir = float3(-1.0,  -ndc.y,  ndc.x); break; // -X
        case 2: dir = float3( ndc.x,  1.0,   ndc.y); break; // +Y
        case 3: dir = float3( ndc.x, -1.0,  -ndc.y); break; // -Y
        case 4: dir = float3( ndc.x, -ndc.y,  1.0);  break; // +Z
        case 5: dir = float3(-ndc.x, -ndc.y, -1.0);  break; // -Z
        default: dir = float3(0, 1, 0); break;
    }
    return normalize(dir);
}

bool ray_sphere_intersect(float3 origin, float3 dir, float radius, out float t0, out float t1)
{
    float a = dot(dir, dir);
    float b = 2.0 * dot(dir, origin);
    float c = dot(origin, origin) - radius * radius;
    float disc = b * b - 4.0 * a * c;
    if (disc < 0.0) { t0 = t1 = 0.0; return false; }
    float sq = sqrt(disc);
    t0 = (-b - sq) / (2.0 * a);
    t1 = (-b + sq) / (2.0 * a);
    return true;
}

float rayleigh_phase(float cos_theta)
{
    return (3.0 / (16.0 * 3.14159265)) * (1.0 + cos_theta * cos_theta);
}

float mie_phase(float cos_theta, float g)
{
    float g2 = g * g;
    return (3.0 / (8.0 * 3.14159265)) *
           ((1.0 - g2) * (1.0 + cos_theta * cos_theta)) /
           ((2.0 + g2) * pow(abs(1.0 + g2 - 2.0 * g * cos_theta), 1.5));
}

float3 compute_sky(float3 ray_dir, float3 sun_dir, float sun_intensity)
{
    float3 ray_origin = float3(0.0, k_earth_radius + k_ray_origin_height, 0.0);

    // Find atmosphere exit
    float t0, t1;
    if (!ray_sphere_intersect(ray_origin, ray_dir, k_atmosphere_radius, t0, t1))
        return float3(0.0);

    float t_start = max(t0, 0.0);
    float t_end   = t1;

    float step_size = (t_end - t_start) / float(k_view_samples);

    float3 total_rayleigh = float3(0.0);
    float3 total_mie      = float3(0.0);

    float optical_depth_rayleigh = 0.0;
    float optical_depth_mie      = 0.0;

    float cos_theta = dot(ray_dir, sun_dir);

    for (int i = 0; i < k_view_samples; ++i)
    {
        float3 sample_pos = ray_origin + ray_dir * (t_start + (float(i) + 0.5) * step_size);
        float  height     = length(sample_pos) - k_earth_radius;

        float hr = exp(-height / k_h_rayleigh) * step_size;
        float hm = exp(-height / k_h_mie)      * step_size;

        optical_depth_rayleigh += hr;
        optical_depth_mie      += hm;

        // Secondary ray toward sun
        float t_sun0, t_sun1;
        ray_sphere_intersect(sample_pos, sun_dir, k_atmosphere_radius, t_sun0, t_sun1);
        float sun_step = t_sun1 / float(k_light_samples);

        float sun_optical_depth_rayleigh = 0.0;
        float sun_optical_depth_mie      = 0.0;
        bool  blocked                    = false;

        for (int j = 0; j < k_light_samples; ++j)
        {
            float3 sun_sample_pos = sample_pos + sun_dir * ((float(j) + 0.5) * sun_step);
            float  sun_height     = length(sun_sample_pos) - k_earth_radius;

            if (sun_height < 0.0) { blocked = true; break; }

            sun_optical_depth_rayleigh += exp(-sun_height / k_h_rayleigh) * sun_step;
            sun_optical_depth_mie      += exp(-sun_height / k_h_mie)      * sun_step;
        }

        if (blocked) continue;

        float3 tau = k_beta_rayleigh * (optical_depth_rayleigh + sun_optical_depth_rayleigh)
                   + k_beta_mie * 1.1 * (optical_depth_mie + sun_optical_depth_mie);

        float3 attenuation = exp(-tau);

        total_rayleigh += attenuation * hr;
        total_mie      += attenuation * hm;
    }

    float3 result = sun_intensity * (
        rayleigh_phase(cos_theta) * k_beta_rayleigh * total_rayleigh +
        mie_phase(cos_theta, k_g) * k_beta_mie      * total_mie);

    return result;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void generate_sky_cs(
    uint3 dispatch_id : SV_DispatchThreadID,
    uniform AtmospherePushConstants pc)
{
    if (dispatch_id.x >= pc.face_size || dispatch_id.y >= pc.face_size)
        return;

    float2 uv  = (float2(dispatch_id.xy) + 0.5) / float(pc.face_size);
    float3 dir = face_uv_to_direction(pc.output_cubemap_face, uv);

    float3 color = compute_sky(dir, normalize(pc.sun_direction), pc.sun_intensity);

    // Tonemap slightly to avoid extreme fireflies at sun disk,
    // but keep HDR range — tonemapping proper happens later
    color = color / (color + 1.0); // Reinhard just for firefly suppression
    color = color * 2.0;           // re-expose back into HDR range

    get_storage_image(pc.output_image_index)[dispatch_id.xy] = float4(color, 1.0);
}