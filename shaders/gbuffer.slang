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
    float3 albedo;
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
    if ((mat.flags & FLAG_ALBEDO_MAP) != 0)
    {
        float4 a = sample_texture(uv, mat.albedo_map, sampler_index);
        surf.albedo = a.rgb * mat.albedo_factor.rgb;
    }
    else
    {
        surf.albedo = mat.albedo_factor.rgb;
    }

    // Normal (tangent-space -> world-space)
    float3 N = safe_normalize(n_ws_geom);
    if ((mat.flags & FLAG_NORMAL_MAP) != 0)
    {
        float3 n_ts = sample_normal_map_ts(uv, mat.normal_map, sampler_index);
        N = normal_ts_to_ws(n_ts, t_ws, b_ws, N);
    }
    surf.normal_ws = N;

    // Roughness
    if ((mat.flags & FLAG_ROUGHNESS_MAP) != 0)
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
    if ((mat.flags & FLAG_METALLIC_MAP) != 0)
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
    if ((mat.flags & FLAG_OCCLUSION_MAP) != 0)
    {
        surf.occlusion = sample_texture(uv, mat.occlusion_map, sampler_index).r;
    }
    else
    {
        surf.occlusion = 1.0;
    }

    // Emissive
    if ((mat.flags & FLAG_EMISSIVE_MAP) != 0)
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

    float3 f0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

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
    float3 L = -light_dir_from;
    float3 radiance = light_color * light_intensity;
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

// -------------------------------------------------------------------------------------------------
// GBuffer pass (VS+FS)
// GBuffer targets (you stated):
//   gbuffer0: ALBEDO_AO           VK_FORMAT_R8G8B8A8_UNORM
//   gbuffer1: NORMAL_OCT + RM     VK_FORMAT_R16G16B16A16_UNORM
//   gbuffer2: EMISSIVE_HDR        VK_FORMAT_R16G16B16A16_SFLOAT
// Depth: D32 (sampled via a view you can index in sampled_images[])
// -------------------------------------------------------------------------------------------------
struct VertexInput
{
    float3 position;
    uint   normal;      // 10_10_10_2 snorm-ish
    uint   uvs;         // 8_8_8_8 unorm
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
/*
struct RenderingPushConstants {
    const DeviceAddress ubo;
    const DeviceAddress transforms;
    const DeviceAddress draw_material_ids;
    const DeviceAddress materials;
    const u32 base_draw_id;
    const u32 sampler_index;};
*/

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

    // Transform basis by model (assumes no crazy non-uniform scale; if you have it, use inverse-transpose for normal)
    float3x3 m3 = (float3x3)model;
    o.n_ws = safe_normalize(mul(m3, n_local));
    o.t_ws = safe_normalize(mul(m3, t_local));
    o.b_ws = safe_normalize(mul(m3, b_local));

    UV4 uvpack = unpack_uvs_unorm4x8(input.uvs);
    o.uv = uvpack.uv0;
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

    // RT0: albedo (linear) + AO
    out_gbuf.albedo_ao = float4(surf.albedo, saturate(surf.occlusion));

    // RT1: oct normal in [0,1] + roughness + metallic
    float2 oct = oct_encode(surf.normal_ws);
    float2 oct_u = oct_to_unorm(oct);
    out_gbuf.normal_rough_metal = float4(oct_u, surf.roughness, surf.metallic);

    // RT2: emissive HDR
    out_gbuf.emissive_hdr = float4(surf.emissive, 1.0);
}

// -------------------------------------------------------------------------------------------------
// Deferred lighting compute
// Reads: gbuffer0/1/2 + depth (sampled)
// Writes: lit_hdr_uav (storage image)
// -------------------------------------------------------------------------------------------------
struct DeferredLightingPushConstants
{
    UBO*        frame_ubo;

    PointLight* point_lights;        // use your compact list if you want
    uint*       point_light_count;

    uint gbuffer0_index;
    uint gbuffer1_index;
    uint gbuffer2_index;
    uint depth_index;

    uint lit_hdr_uav_index;

    uint sampler_index;              // for reading gbuffer + depth
};

float3 reconstruct_world_pos_from_depth(float2 uv, float depth, UBO* ubo)
{
    // uv [0,1] -> ndc [-1,1]
    float2 ndc_xy = uv * 2.0 - 1.0;

    // Vulkan depth is [0,1]. Works with reverse-Z as long as inv_view_projection matches.
    float4 clip = float4(ndc_xy, depth, 1.0);
    float4 world = mul(ubo->inv_view_projection, clip);
    return world.xyz / max(world.w, 1e-6);
}

struct VSOut
{
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
};

[shader("vertex")]
VSOut vs_fullscreen_main(uint vertex_id : SV_VertexID)
{
    // Fullscreen triangle in clip space
    float2 pos = float2(
        (vertex_id == 2) ?  3.0 : -1.0,
        (vertex_id == 1) ?  3.0 : -1.0
    );

    VSOut o;
    o.position = float4(pos, 0.0, 1.0);

    // Clip [-1,1] -> UV [0,1]
    // Flip Y for Vulkan-style UV orientation (match your other sampling expectations)
    o.uv = pos * 0.5 + 0.5;
    o.uv.y = 1.0 - o.uv.y;

    return o;
}

[shader("fragment")]
float4 fs_fullscreen_main(VSOut i, uniform DeferredLightingPushConstants pc) : SV_Target0
{
    // Use the VS-provided UVs (these are the "correct uvs from the vertex shader")
    float2 uv = i.uv;

    // Same logic as compute shader, but writing SV_Target0 instead of RWTexture2D.
    float4 g0 = sample_texture(uv, pc.gbuffer0_index, pc.sampler_index);
    float4 g1 = sample_texture(uv, pc.gbuffer1_index, pc.sampler_index);
    float4 g2 = sample_texture(uv, pc.gbuffer2_index, pc.sampler_index);

    // Depth sampled as .x (requires a sampled view for the depth image compatible with your sample_texture path)
    float depth = sample_texture(uv, pc.depth_index, pc.sampler_index).x;

    // Reverse-Z convention: you cleared predepth to 0.0 -> treat as "no geometry"
    if (depth <= 0.0)
        return float4(0.0, 0.0, 0.0, 1.0);

    float3 albedo = g0.rgb;
    float  ao     = saturate(g0.a);

    float2 oct_u  = g1.xy;
    float  rough  = clamp(g1.z, 0.04, 1.0);
    float  metal  = saturate(g1.w);

    float3 emissive = g2.rgb;

    float3 N = oct_decode(unorm_to_oct(oct_u));
    float3 world_pos = reconstruct_world_pos_from_depth(uv, depth, pc.frame_ubo);
    float3 V = safe_normalize(pc.frame_ubo->camera_position.xyz - world_pos);

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
        pc.frame_ubo->sun_intensity,
        float3(1.0, 1.0, 1.0),
        V,
        N,
        albedo,
        rough,
        metal
    );

    // Point lights
    uint light_count = *pc.point_light_count;
    for (uint li = 0; li < light_count; ++li)
    {
        color += evaluate_point_light(
            pc.point_lights[li],
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
[shader("compute")]
[numthreads(32, 32, 1)]
void main_gbuffer_lighting(uint3 dtid : SV_DispatchThreadID, uniform DeferredLightingPushConstants pc)
{
    RWTexture2D<float4> out_img = storage_images[NonUniformResourceIndex(pc.lit_hdr_uav_index)];

    uint width = 0, height = 0;
    out_img.GetDimensions(width, height);

    if (dtid.x >= width || dtid.y >= height)
        return;

    float2 uv = (float2(dtid.xy) + 0.5) / float2(width, height);

    float4 g0 = sample_texture(uv, pc.gbuffer0_index, pc.sampler_index);
    float4 g1 = sample_texture(uv, pc.gbuffer1_index, pc.sampler_index);
    float4 g2 = sample_texture(uv, pc.gbuffer2_index, pc.sampler_index);

    // Depth sampled as .x. You must provide a sampled view compatible with Texture2D<float4>.
    float depth = sample_texture(uv, pc.depth_index, pc.sampler_index).x;

    // Reverse-Z convention: you cleared predepth to 0.0 -> treat as "no geometry"
    if (depth <= 0.0)
    {
        out_img[dtid.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float3 albedo = g0.rgb;
    float  ao     = saturate(g0.a);

    float2 oct_u  = g1.xy;
    float  rough  = clamp(g1.z, 0.04, 1.0);
    float  metal  = saturate(g1.w);

    float3 emissive = g2.rgb;

    float3 N = oct_decode(unorm_to_oct(oct_u));
    float3 world_pos = reconstruct_world_pos_from_depth(uv, depth, pc.frame_ubo);

    float3 V = safe_normalize(pc.frame_ubo->camera_position.xyz - world_pos);

    float3 color = calculate_ambient(
        albedo,
        metal,
        ao,
        float3(0.03, 0.03, 0.03),
        0.1 //pc.ambient_intensity
    );

    // Sun light
    color += evaluate_directional_light(
        pc.frame_ubo->sun_dir,
        pc.frame_ubo->sun_intensity,
        float3(1.0, 1.0, 1.0),
        V,
        N,
        albedo,
        rough,
        metal
    );

    // Point lights (assumes pc.point_lights is your compacted list)
    uint light_count = *pc.point_light_count;
    for (uint i = 0; i < light_count; ++i)
    {
        color += evaluate_point_light(
            pc.point_lights[i],
            world_pos,
            V,
            N,
            albedo,
            rough,
            metal
        );
    }

    color += emissive;

    out_img[dtid.xy] = float4(color, 1.0);
}
