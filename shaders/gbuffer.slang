import common;

float3 safe_normalize(float3 v)
{
    float l2 = dot(v, v);
    if (l2 <= 1e-20) return float3(0.0, 1.0, 0.0);
    return v * rsqrt(l2);
}

// -------------------------------------------------------------------------------------------------
// Oct normal encoding for GBuffer
// -------------------------------------------------------------------------------------------------
float2 oct_encode(float3 n)
{
    n = safe_normalize(n);
    float inv_l1 = 1.0 / (abs(n.x) + abs(n.y) + abs(n.z) + 1e-8);
    n *= inv_l1;

    float2 p = n.xy;
    if (n.z < 0.0)
    {
        p = (1.0 - abs(p.yx)) * (float2(p.x >= 0 ? 1 : -1, p.y >= 0 ? 1 : -1));
    }
    return p; // [-1,1]
}

float3 oct_decode(float2 e)
{
    float3 n = float3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    float t = saturate(-n.z);
    n.xy += select(n.xy >= 0.0, -t, t);

    return safe_normalize(n);
}

float2 oct_to_unorm(float2 oct) { return oct * 0.5 + 0.5; }
float2 unorm_to_oct(float2 u)   { return u * 2.0 - 1.0; }

// -------------------------------------------------------------------------------------------------
// Material evaluation (NOW uses vertex T/B/N for normal mapping)
// -------------------------------------------------------------------------------------------------
struct SurfaceMaterial
{
    float4 albedo;
    float3 normal_ws;
    float  roughness;
    float  metallic;
    float  occlusion;
    float3 emissive;
};

float3 sample_normal_map_ts(float2 uv, uint normal_map_index, uint sampler_index)
{
    float3 n = sample_texture(uv, normal_map_index, sampler_index).xyz;
    // Tangent-space normal map in [0,1] -> [-1,1]
    n = n * 2.0 - 1.0;
    return safe_normalize(n);
}

float3 normal_ts_to_ws(float3 n_ts, float3 t_ws, float3 b_ws, float3 n_ws)
{
    // Build TBN from interpolated basis.
    // We assume your mesh tangent/bitangent were orthonormalized on CPU (your compute_tangent_basis does).
    float3 T = safe_normalize(t_ws);
    float3 B = safe_normalize(b_ws);
    float3 N = safe_normalize(n_ws);

    // If the basis drifts under interpolation, re-orthonormalize lightly:
    T = safe_normalize(T - N * dot(N, T));
    B = safe_normalize(B - N * dot(N, B));

    // Slang mul convention: mul(vector, matrix) is vector * matrix.
    float3x3 TBN = float3x3(T, B, N);
    return safe_normalize(mul(n_ts, TBN));
}

SurfaceMaterial evaluate_material_tbn(
    Material mat,
    float2 uv,
    float3 n_ws_geom,
    float3 t_ws,
    float3 b_ws,
    uint sampler_index)
{
    SurfaceMaterial surf;

    // Albedo
    if (mat.has_albedo_map())
    {
        float4 a = sample_texture(uv, mat.albedo_map, sampler_index);
        surf.albedo.rgb = a.rgb * mat.albedo_factor.rgb;
        surf.albedo.a = 1.0F;
    }
    else
    {
        surf.albedo = mat.albedo_factor.rgba;
    }

    // Normal (tangent-space -> world-space)
    float3 N = safe_normalize(n_ws_geom);
    if (mat.has_normal_map())
    {
        float3 n_ts = sample_normal_map_ts(uv, mat.normal_map, sampler_index);
        N = normal_ts_to_ws(n_ts, t_ws, b_ws, N);
    }
    surf.normal_ws = N;

    // Roughness
    if (mat.has_roughness_map())
    {
        float r = sample_texture(uv, mat.roughness_map, sampler_index).r;
        surf.roughness = r * mat.roughness_factor;
    }
    else
    {
        surf.roughness = mat.roughness_factor;
    }
    surf.roughness = clamp(surf.roughness, 0.04, 1.0);

    // Metallic
    if (mat.has_metallic_map())
    {
        float m = sample_texture(uv, mat.metallic_map, sampler_index).r;
        surf.metallic = m * mat.metallic_factor;
    }
    else
    {
        surf.metallic = mat.metallic_factor;
    }
    surf.metallic = saturate(surf.metallic);

    // Occlusion
    if (mat.has_occlusion_map())
    {
        surf.occlusion = sample_texture(uv, mat.occlusion_map, sampler_index).r;
    }
    else
    {
        surf.occlusion = 1.0;
    }

    // Emissive
    if (mat.has_emissive_map())
    {
        float3 e = sample_texture(uv, mat.emissive_map, sampler_index).rgb;
        surf.emissive = e * mat.emissive_factor;
    }
    else
    {
        surf.emissive = mat.emissive_factor;
    }

    return surf;
}

// -------------------------------------------------------------------------------------------------
// PBR lighting (same model you already use)
// -------------------------------------------------------------------------------------------------
static const float PI = 3.14159265359;

float3 fresnel_schlick(float cos_theta, float3 f0)
{
    return f0 + (1.0 - f0) * pow(saturate(1.0 - cos_theta), 5.0);
}

float distribution_ggx(float3 N, float3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = saturate(dot(N, H));
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return num / max(denom, 0.0001);
}

float geometry_schlick_ggx(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    return num / max(denom, 0.0001);
}

float geometry_smith(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = saturate(dot(N, V));
    float NdotL = saturate(dot(N, L));
    float ggx2 = geometry_schlick_ggx(NdotV, roughness);
    float ggx1 = geometry_schlick_ggx(NdotL, roughness);
    return ggx1 * ggx2;
}

float3 calculate_direct_lighting(
    float3 L,
    float3 V,
    float3 N,
    float3 albedo,
    float  roughness,
    float  metallic,
    float3 radiance)
{
    float3 H = safe_normalize(V + L);

    float3 f0 = lerp(float3(0.04), albedo, metallic);

    float NDF = distribution_ggx(N, H, roughness);
    float G   = geometry_smith(N, V, L, roughness);
    float3 F  = fresnel_schlick(saturate(dot(H, V)), f0);

    float3 numerator = NDF * G * F;
    float denom = 4.0 * max(saturate(dot(N, V)), 1e-4) * max(saturate(dot(N, L)), 1e-4);
    float3 specular = numerator / max(denom, 0.001);

    float3 kS = F;
    float3 kD = (1.0 - kS) * (1.0 - metallic);

    float NdotL = saturate(dot(N, L));
    return (kD * albedo / PI + specular) * radiance * NdotL;
}

float3 evaluate_directional_light(
    float3 light_dir_from,
    float  light_intensity,
    float3 light_color,
    float3 V,
    float3 N,
    float3 albedo,
    float  roughness,
    float  metallic)
{
    float3 L = light_dir_from;
    float3 radiance = light_color * light_intensity;
    return calculate_direct_lighting(L, V, N, albedo, roughness, metallic, radiance);
}

float3 evaluate_point_light_fixed_v3(
    PointLight light,
    float3 world_pos,
    float3 V,
    float3 N,
    float3 albedo,
    float  roughness,
    float  metallic)
{
    float3 light_pos = light.position_radius.xyz;
    float  radius    = light.position_radius.w;
    float3 light_col = light.colour_intensity.rgb;
    float  intensity = light.colour_intensity.a;

    float3 L = light_pos - world_pos;
    float dist_sq = dot(L, L);
    float dist = sqrt(dist_sq);

    if (dist < 1e-4) return 0.0.xxx;
    L /= dist;

    // Early out if beyond radius
    if (dist > radius) return 0.0.xxx;

    // UE4-style windowed attenuation
    float radius_sq = radius * radius;
    float attenuation = saturate(1.0 - (dist_sq / radius_sq));
    attenuation = attenuation * attenuation; // Square for smooth falloff
    attenuation = attenuation / (dist_sq + 1.0); // Physical falloff

    float3 radiance = light_col * intensity * attenuation;
    return calculate_direct_lighting(L, V, N, albedo, roughness, metallic, radiance);
}

float3 evaluate_point_light(
    PointLight light,
    float3 world_pos,
    float3 V,
    float3 N,
    float3 albedo,
    float  roughness,
    float  metallic)
{
    float3 light_pos = light.position_radius.xyz;
    float  radius    = light.position_radius.w;
    float3 light_col = light.colour_intensity.rgb;
    float  intensity = light.colour_intensity.a;

    float3 L = light_pos - world_pos;
    float dist = length(L);
    if (dist < 1e-4) return 0.0.xxx;
    L /= dist;

    float attenuation = 1.0 / (dist * dist);
    float falloff = 1.0 - smoothstep(radius * 0.75, radius, dist);
    attenuation *= falloff;

    float3 radiance = light_col * intensity * attenuation;
    return calculate_direct_lighting(L, V, N, albedo, roughness, metallic, radiance);
}

float3 calculate_ambient(
    float3 albedo,
    float  metallic,
    float  occlusion,
    float3 ambient_color,
    float  ambient_intensity)
{
    float3 f0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);
    float3 kD = (1.0 - f0) * (1.0 - metallic);
    return kD * albedo * ambient_color * ambient_intensity * occlusion;
}

struct VertexInput
{
    float3 position;
    float2   uvs;
    uint   normal;      // 10_10_10_2 snorm-ish
    uint   tangent;     // 10_10_10_2
    uint   bitangent;   // 10_10_10_2
};

struct VertexOutput
{
    float4 position  : SV_Position;

    float3 world_pos : TEXCOORD0;
    float3 n_ws      : TEXCOORD1;
    float2 uv        : TEXCOORD2;

    float3 t_ws      : TEXCOORD3;
    float3 b_ws      : TEXCOORD4;

    nointerpolation uint material : TEXCOORD5;
};

struct RenderingPushConstants
{
    UBO*       frame_ubo;
    Transform* transforms;
    uint*      draw_material_ids;
    Material*  materials;
    uint       base_draw_id;
    uint       sampler_index; // choose which sampler to use for material textures
};

[shader("vertex")]
VertexOutput main_vs_mdi(
    VertexInput input,
    uint draw_id : SV_DrawIndex,
    uint instance_id : SV_InstanceID,
    uint base_instance : SV_StartInstanceLocation,
    uniform RenderingPushConstants pc)
{
    VertexOutput o = {};

    uint global_draw = pc.base_draw_id + draw_id;
    uint material_id = pc.draw_material_ids[global_draw];
    uint instance_index = instance_id + base_instance;

    float3x4 model_3x4 = pc.transforms[instance_index].m;

    float4x4 model = float4x4(
        model_3x4._m00_m01_m02_m03,
        model_3x4._m10_m11_m12_m13,
        model_3x4._m20_m21_m22_m23,
        float4(0, 0, 0, 1)
    );

    float4 world_pos = mul(model, float4(input.position, 1.0));
    o.position  = mul(pc.frame_ubo->view_projection, world_pos);
    o.world_pos = world_pos.xyz;

    float3 n_local = safe_normalize(unpackSnorm3x10_1x2(input.normal).xyz);
    float3 t_local = safe_normalize(unpackSnorm3x10_1x2(input.tangent).xyz);
    float3 b_local = safe_normalize(unpackSnorm3x10_1x2(input.bitangent).xyz);

    float3x3 m3 = (float3x3)model;
    o.n_ws = safe_normalize(mul(m3, n_local));
    o.t_ws = safe_normalize(mul(m3, t_local));
    o.b_ws = safe_normalize(mul(m3, b_local));

    o.uv = input.uvs;
    o.material = material_id;

    return o;
}

struct GBufferOutput
{
    float4 albedo_ao          : SV_Target0;
    float4 normal_rough_metal : SV_Target1;
    float4 emissive_hdr       : SV_Target2;
};

[shader("fragment")]
void main_fs_mdi(
    VertexOutput input,
    uniform RenderingPushConstants pc,
    out GBufferOutput out_gbuf)
{
   Material mat = pc.materials[input.material];

    SurfaceMaterial surf = evaluate_material_tbn(
        mat,
        input.uv,
        input.n_ws,
        input.t_ws,
        input.b_ws,
        pc.sampler_index
    );

    out_gbuf.albedo_ao = float4(surf.albedo.rgb, saturate(surf.occlusion));

    float2 oct = oct_encode(surf.normal_ws);
    float2 oct_u = oct;
    out_gbuf.normal_rough_metal = float4(oct_u, surf.roughness, surf.metallic);

    out_gbuf.emissive_hdr = float4(surf.emissive, float(input.material));
}

// -------------------------------------------------------------------------------------------------
// CLUSTERED DEFERRED LIGHTING
// -------------------------------------------------------------------------------------------------

struct Cluster
{
    uint light_offset;
    uint light_count;
};

struct DeferredLightingPushConstants
{
    UBO*        frame_ubo;
    PointLight* point_lights;
    float4x4 shadow_matrix;
    uint tiles_x;
    uint tiles_y;
    uint tiles_z;
    float log_z_scale;
    Cluster* clusters;               // Cluster offset/count data
    uint*    cluster_light_indices;  // Packed light indices
    uint gbuffer0_index;
    uint gbuffer1_index;
    uint gbuffer2_index;
    uint depth_index;
    uint debug_culling_index;
    uint lit_hdr_uav_index;
    uint sampler_index;
    uint shadow_texture_index;
    uint shadow_sampler_index;
    uint debug_mode;
};

uint compute_cluster_index(
    float2 screen_pos,
    float view_z,
    float z_near,
    float log_z_scale,
    uint tiles_x,
    uint tiles_y,
    uint tiles_z,
    float2 screen_size)
{
    float2 tile_coord = screen_pos / screen_size;
    uint cluster_x = uint(tile_coord.x * float(tiles_x));
    uint cluster_y = uint(tile_coord.y * float(tiles_y));

    cluster_x = min(cluster_x, tiles_x - 1);
    cluster_y = min(cluster_y, tiles_y - 1);

    float depth_ratio = abs(view_z) / z_near;
    uint cluster_z = uint(log2(max(depth_ratio, 1.0)) * log_z_scale);  // ✅ Clamp to >= 1.0
    cluster_z = min(cluster_z, tiles_z - 1);

    return cluster_x + cluster_y * tiles_x + cluster_z * tiles_x * tiles_y;
}

float3 reconstruct_world_pos_from_depth_fixed(float2 uv, float depth, UBO* ubo)
{
    float2 ndc_xy = uv * 2.0 - 1.0;
    
    float4 clip = float4(ndc_xy, depth, 1.0);
    float4 world = mul(ubo->inv_view_projection, clip);
    
    return world.xyz / world.w;
}

struct VSOut
{
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
};


float calculate_shadow(float3 world_pos, float4x4 shadow_matrix, uint tex_idx, uint comp_samp_idx)
{
    // 1. Transform world position to Light Clip Space
    float4 shadow_coord = mul(shadow_matrix, float4(world_pos, 1.0));
    
    // 2. Perspective divide to get NDC
    float3 ndc = shadow_coord.xyz / shadow_coord.w;

    // 3. Transform NDC [-1, 1] to UV [0, 1]
    // Vulkan Y is pointing down, so we flip the Y calculation here 
    // to match how textures are stored/sampled.
    float2 shadow_uv = float2(
        ndc.x * 0.5 + 0.5,
        ndc.y * -0.5 + 0.5 // Note the negative 0.5 to flip Y
    );

    // 4. Boundary Check
    // If we are outside the light's frustum, the pixel is lit.
    if (any(shadow_uv < 0.0) || any(shadow_uv > 1.0) || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }

    uint t_idx = NonUniformResourceIndex(tex_idx);
    uint cs_idx = NonUniformResourceIndex(comp_samp_idx);

    // 5. Reverse-Z Bias
    // In Reverse-Z, higher values are closer to the light.
    // We add a small bias to the current pixel's depth to avoid self-shadowing (acne).
    float bias = 0.0005; 
    float reference_depth = ndc.z + bias;

    // 6. Hardware PCF
    // This uses your VK_COMPARE_OP_GREATER_OR_EQUAL sampler.
    return sampled_images[t_idx].SampleCmp(
        comparison_samplers[cs_idx],
        shadow_uv,
        reference_depth
    );

}
[shader("fragment")]
float4 fs_fullscreen_main(VSOut i, uniform DeferredLightingPushConstants pc) : SV_Target0
{
    float2 uv = i.uv;

    // Sample G-Buffer
    float4 g0 = sample_texture(uv, pc.gbuffer0_index, pc.sampler_index);
    float4 g1 = sample_texture(uv, pc.gbuffer1_index, pc.sampler_index);
    float4 g2 = sample_texture(uv, pc.gbuffer2_index, pc.sampler_index);
    float depth = sample_texture(uv, pc.depth_index, pc.sampler_index).x;

    // Early out if no geometry
    if (depth <= 0.0)
        return float4(0.0, 0.0, 0.0, 1.0);

    // Unpack G-Buffer
    float3 albedo = g0.rgb;
    float  ao     = saturate(g0.a);

    float2 oct_u  = g1.xy;
    float  rough  = clamp(g1.z, 0.04, 1.0);
    float  metal  = saturate(g1.w);

    float3 emissive = g2.rgb;

    float3 N = oct_decode(oct_u);
    float3 world_pos = reconstruct_world_pos_from_depth_fixed(uv, depth, pc.frame_ubo);
    float shadow_factor = calculate_shadow(
    world_pos,
    pc.shadow_matrix,
    pc.shadow_texture_index,
    pc.shadow_sampler_index
);
    float3 V = safe_normalize(pc.frame_ubo->camera_position.xyz - world_pos);

    // Ambient
    float3 color = calculate_ambient(
        albedo,
        metal,
        ao,
        float3(0.03, 0.03, 0.03),
        0.1
    );

    // Sun light
    color += evaluate_directional_light(
        pc.frame_ubo->sun_dir,
        pc.frame_ubo->sun_intensity * shadow_factor,
        float3(1.0, 1.0, 1.0),
        V,
        N,
        albedo,
        rough,
        metal
    ) ;

    float3 view_pos = mul(pc.frame_ubo->view, float4(world_pos, 1.0)).xyz;
    float2 screen_size = pc.frame_ubo->viewport_size;

    uint cluster_idx = compute_cluster_index(
        i.position.xy,
        view_pos.z,
        0.1F, // pc.z_near some time in the future?
        pc.log_z_scale,
        pc.tiles_x,
        pc.tiles_y,
        pc.tiles_z,
        screen_size
    );

    var cluster = pc.clusters[cluster_idx];

    const uint MAX_LIGHTS_PER_PIXEL = 128u; // Tune this!
uint lights_to_process = min(cluster.light_count, MAX_LIGHTS_PER_PIXEL);
    for (uint li = 0; li < lights_to_process; ++li)
    {
        uint light_idx = pc.cluster_light_indices[cluster.light_offset + li];
        PointLight light = pc.point_lights[light_idx];

        color += evaluate_point_light_fixed_v3(
            light,
            world_pos,
            V,
            N,
            albedo,
            rough,
            metal
        );
    }

    color += emissive;

    return float4(color, 1.0);
}
