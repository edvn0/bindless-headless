import common;
import maths;
import pointer;
import constants;
import dithering;

// -------------------------------------------------------------------------------------------------
// Material evaluation (NOW uses vertex T/B/N for normal mapping)
// -------------------------------------------------------------------------------------------------
struct SurfaceMaterial {
    float4 albedo;
    float3 normal_ws;
    float roughness;
    float metallic;
    float occlusion;
    float3 emissive;
};

float3 sample_normal_map_ts(float2 uv, uint normal_map_index, uint sampler_index) {
    float4 s = sample_texture(uv, normal_map_index, sampler_index);
    float2 nml = s.xy; // X=R, Y=G
    nml = nml * 2.0 - 1.0;
    float z = sqrt(saturate(1.0 - dot(nml, nml)));
    return float3(nml.x, nml.y, z);
}

float3 normal_ts_to_ws(float3 n_ts, float3 t_ws, float3 b_ws, float3 n_ws) {
    float3 N = safe_normalize(n_ws);
    float3 T = safe_normalize(t_ws - N * dot(N, t_ws));

    // Preserve handedness from the original bitangent
    float3 B = cross(N, T);
    float handedness = dot(B, b_ws) < 0.0 ? -1.0 : 1.0;
    B *= handedness;

    float3x3 TBN = float3x3(T, B, N);
    return safe_normalize(mul(n_ts, TBN));
}

SurfaceMaterial evaluate_material_tbn(Material mat, float4 albedo_sample, float4 multi_uv, float3 n_ws_geom, float3 t_ws, float3 b_ws,
                                      uint sampler_index) {
    SurfaceMaterial surf;
    var uv = multi_uv.xy;

    surf.albedo.rgb = albedo_sample.rgb * mat.albedo_factor.rgb;
    surf.albedo.a = 1.0;

    float3 N = safe_normalize(n_ws_geom);
    if (mat.has_normal_map()) {
        float3 n_ts = sample_normal_map_ts(uv, mat.normal_map, sampler_index);
        N = normal_ts_to_ws(n_ts, t_ws, b_ws, N);
    }
    surf.normal_ws = N;

    // Roughness
    if (mat.has_roughness_map()) {
        float r = sample_texture(uv, mat.roughness_map, sampler_index).g;
        surf.roughness = r * mat.roughness_factor;
    } else {
        surf.roughness = mat.roughness_factor;
    }
    surf.roughness = clamp(surf.roughness, 0.04, 1.0);

    // Metallic
    if (mat.has_metallic_map()) {
        float m = sample_texture(uv, mat.metallic_map, sampler_index).b;
        surf.metallic = m * mat.metallic_factor;
    } else {
        surf.metallic = mat.metallic_factor;
    }
    surf.metallic = saturate(surf.metallic);

    // Occlusion
    if (mat.has_occlusion_map()) {
        surf.occlusion = sample_texture(uv, mat.occlusion_map, sampler_index).r;
    } else {
        surf.occlusion = 1.0;
    }

    // Emissive
    if (mat.has_emissive_map()) {
        float3 e = sample_texture(uv, mat.emissive_map, sampler_index).rgb;
        surf.emissive = e * mat.emissive_factor;
    } else {
        surf.emissive = mat.emissive_factor;
    }

    return surf;
}

static const float PI = 3.14159265359;

float3 fresnel_schlick(float cos_theta, float3 f0) { return f0 + (1.0 - f0) * pow(saturate(1.0 - cos_theta), 5.0); }

float distribution_ggx(float3 N, float3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = saturate(dot(N, H));
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return num / max(denom, 0.0001);
}

float geometry_schlick_ggx(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    return num / max(denom, 0.0001);
}

float geometry_smith(float3 N, float3 V, float3 L, float roughness) {
    float NdotV = saturate(dot(N, V));
    float NdotL = saturate(dot(N, L));
    float ggx2 = geometry_schlick_ggx(NdotV, roughness);
    float ggx1 = geometry_schlick_ggx(NdotL, roughness);
    return ggx1 * ggx2;
}

float3 calculate_direct_lighting(float3 L, float3 V, float3 N, float3 albedo, float roughness, float metallic,
                                 float3 radiance) {
    float3 H = safe_normalize(V + L);

    float3 f0 = lerp(float3(0.04), albedo, metallic);

    float NDF = distribution_ggx(N, H, roughness);
    float G = geometry_smith(N, V, L, roughness);
    float3 F = fresnel_schlick(saturate(dot(H, V)), f0);

    float3 numerator = NDF * G * F;
    float denom = 4.0 * max(saturate(dot(N, V)), 1e-4) * max(saturate(dot(N, L)), 1e-4);
    float3 specular = numerator / max(denom, 0.001);

    float3 kS = F;
    float3 kD = (1.0 - kS) * (1.0 - metallic);

    float NdotL = saturate(dot(N, L));
    return (kD * albedo / PI + specular) * radiance * NdotL;
}

float3 evaluate_directional_light(float3 light_dir_from, float light_intensity, float3 light_color, float3 V, float3 N,
                                  float3 albedo, float roughness, float metallic) {
    float3 L = light_dir_from;
    float3 radiance = light_color * light_intensity;
    return calculate_direct_lighting(L, V, N, albedo, roughness, metallic, radiance);
}

float3 evaluate_point_light_pbr(PointLight light, float3 world_pos, float3 V, float3 N, float3 albedo, float roughness,
                                float metallic) {
    float3 light_pos = light.position_radius.xyz;
    float radius = light.position_radius.w;
    float3 light_col = light.colour_intensity.rgb;
    float intensity = light.colour_intensity.a;

    float3 L_unnorm = light_pos - world_pos;
    float dist_sq = dot(L_unnorm, L_unnorm);
    float dist = sqrt(dist_sq);

    if (dist < 1e-4 || dist > radius)
        return 0.0.xxx;

    float3 L = L_unnorm / dist;

    static const float ref_dist_sq = 1.0;
    float inv_sq = ref_dist_sq / max(dist_sq, 0.01);

    float radius_sq = radius * radius;
    float ratio_sq = dist_sq / radius_sq;
    float window = saturate(1.0 - ratio_sq * ratio_sq);
    window *= window;

    float3 radiance = light_col * intensity * inv_sq * window;
    return calculate_direct_lighting(L, V, N, albedo, roughness, metallic, radiance);
}

float3 evaluate_point_light_fixed_v4(PointLight light, float3 world_pos, float3 V, float3 N, float3 albedo,
                                     float roughness, float metallic) {
    float3 light_pos = light.position_radius.xyz;
    float radius = light.position_radius.w;
    float3 light_col = light.colour_intensity.rgb;
    float intensity = light.colour_intensity.a;
    float3 L = light_pos - world_pos;
    float dist = length(L);
    if (dist > radius)
        return 0.0.xxx;

    float norm_dist = saturate(dist / (radius * 0.85));
    float attenuation = (1.0 - norm_dist * norm_dist);
    attenuation = attenuation * attenuation;
    float3 radiance = light_col * intensity * attenuation;
    return calculate_direct_lighting(safe_normalize(L), V, N, albedo, roughness, metallic, radiance);
}

float3 evaluate_point_light_fixed_v3(PointLight light, float3 world_pos, float3 V, float3 N, float3 albedo,
                                     float roughness, float metallic) {
    float3 light_pos = light.position_radius.xyz;
    float radius = light.position_radius.w;
    float3 light_col = light.colour_intensity.rgb;
    float intensity = light.colour_intensity.a;

    float3 L = light_pos - world_pos;
    float dist_sq = dot(L, L);
    float dist = sqrt(dist_sq);

    if (dist < 1e-4)
        return 0.0.xxx;
    L /= dist;

    // Early out if beyond radius
    if (dist > radius)
        return 0.0.xxx;

    // UE4-style windowed attenuation
    float radius_sq = radius * radius;
    float attenuation = saturate(1.0 - (dist_sq / radius_sq));
    attenuation = attenuation * attenuation; // Square for smooth falloff
    attenuation = attenuation / (dist_sq + 1.0); // Physical falloff

    float3 radiance = light_col * intensity * attenuation;
    return calculate_direct_lighting(L, V, N, albedo, roughness, metallic, radiance);
}

float3 evaluate_point_light(PointLight light, float3 world_pos, float3 V, float3 N, float3 albedo, float roughness,
                            float metallic) {
    float3 light_pos = light.position_radius.xyz;
    float radius = light.position_radius.w;
    float3 light_col = light.colour_intensity.rgb;
    float intensity = light.colour_intensity.a;

    float3 L = light_pos - world_pos;
    float dist = length(L);
    if (dist < 1e-4)
        return 0.0.xxx;
    L /= dist;

    float attenuation = 1.0 / (dist * dist);
    float falloff = 1.0 - smoothstep(radius * 0.75, radius, dist);
    attenuation *= falloff;

    float3 radiance = light_col * intensity * attenuation;
    return calculate_direct_lighting(L, V, N, albedo, roughness, metallic, radiance);
}

float3 calculate_ambient(float3 albedo, float metallic, float occlusion, float3 ambient_color,
                         float ambient_intensity) {
    float3 f0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);
    float3 kD = (1.0 - f0) * (1.0 - metallic);
    return kD * albedo * ambient_color * ambient_intensity * occlusion;
}

struct VertexInput {
    float3 position;
    uint uv0; // 0xPOS0POS1
    uint uv1; // 0xPOS0POS1
    uint normal; // 10_10_10_2 snorm-ish
    uint tangent; // 10_10_10_2
    uint bitangent; // 10_10_10_2
};

struct VertexOutput {
    float4 position : SV_Position;

    float3 world_pos : TEXCOORD0;
    float3 n_ws : TEXCOORD1;
    float4 uv : TEXCOORD2;

    float3 t_ws : TEXCOORD3;
    float3 b_ws : TEXCOORD4;

    nointerpolation uint material : TEXCOORD5;
};

struct RenderingPushConstants {
    DeviceReadPtr<UBO> frame_ubo;
    DeviceReadPtr<InstanceData> instance_data;
    DeviceReadPtr<uint> draw_material_ids;
    DeviceReadPtr<Material> materials;
    uint base_draw_id;
    uint sampler_index;
};

[shader("vertex")] VertexOutput
main_vs_mdi(VertexInput input, uint draw_id : SV_DrawIndex, uint instance_id : SV_InstanceID,
            uint base_instance : SV_StartInstanceLocation, uniform RenderingPushConstants pc) {
    VertexOutput o = {};

    uint global_draw = pc.base_draw_id + draw_id;
    uint material_id = pc.draw_material_ids[global_draw];
    uint instance_index = instance_id + base_instance;

 float3x4 model_3x4 = pc.instance_data[instance_index].transform.m;
     float4x4 model = float4x4(
         model_3x4._m00_m01_m02_m03,
         model_3x4._m10_m11_m12_m13,
         model_3x4._m20_m21_m22_m23,
         float4(0, 0, 0, 1)
     );

    float4 world_pos = mul(model, float4(input.position, 1.0));
    o.position = mul(pc.frame_ubo->view_projection, world_pos);
    o.world_pos = world_pos.xyz;

    float3 n_local = safe_normalize(unpackSnorm3x10_1x2(input.normal).xyz);
    float3 t_local = safe_normalize(unpackSnorm3x10_1x2(input.tangent).xyz);
    float3 b_local = safe_normalize(unpackSnorm3x10_1x2(input.bitangent).xyz);

    float3x3 m3 = (float3x3) model;
    o.n_ws = safe_normalize(mul(m3, n_local));
    o.t_ws = safe_normalize(mul(m3, t_local));
    o.b_ws = safe_normalize(mul(m3, b_local));

    o.uv = float4(unpackHalf2x16ToFloat(input.uv0), unpackHalf2x16ToFloat(input.uv1));
    o.material = material_id;

    printf("%d", material_id);

    return o;
}

struct GBufferOutput {
    float4 albedo_ao : SV_Target0;
    float4 normal_rough_metal : SV_Target1;
    float4 emissive_hdr : SV_Target2;
};

[shader("fragment")] void main_fs_mdi(VertexOutput input, uniform RenderingPushConstants pc,
                                      bool is_front_face : SV_IsFrontFace, out GBufferOutput out_gbuf) {
    Material mat = pc.materials[NonUniformResourceIndex(input.material)];

        float4 albedo_sample = mat.albedo_factor;
        if (mat.has_albedo_map()) {
            albedo_sample *= sample_texture(input.uv.xy, mat.albedo_map, pc.sampler_index);
        }

        if (mat.is_alpha_tested() && albedo_sample.a < alpha_cutoff) {
            discard;
        }

        if (mat.is_transparent() && is_dithered_discarded(albedo_sample.a, input.position.xy, pc.frame_ubo->frame_index)) {
            discard;
        }

    float3 n_ws = is_front_face ? input.n_ws : -input.n_ws;
    float3 t_ws = is_front_face ? input.t_ws : -input.t_ws;
    float3 b_ws = is_front_face ? input.b_ws : -input.b_ws;
SurfaceMaterial surf = evaluate_material_tbn(mat, albedo_sample, input.uv, n_ws, t_ws, b_ws, pc.sampler_index);

    out_gbuf.albedo_ao = float4(surf.albedo.rgb, saturate(surf.occlusion));

    float2 oct = oct_encode(surf.normal_ws);
    float2 oct_u = oct;
    out_gbuf.normal_rough_metal = float4(oct_u, surf.roughness, surf.metallic);

    out_gbuf.emissive_hdr = float4(surf.emissive, float(input.material));
}

// -------------------------------------------------------------------------------------------------
// CLUSTERED DEFERRED LIGHTING
// -------------------------------------------------------------------------------------------------


/**
struct DeferredLightingPushConstants {
    const DeviceAddress frame_ubo;
    const DeviceAddress point_lights; // All the lights
    const glm::mat4 shadow_matrix;

    u32 tiles_x;
    u32 tiles_y;
    u32 tiles_z;
    float log_z_scale;

    const DeviceAddress clusters;
    const DeviceAddress cluster_light_indices;

    u32 gbuffer0_index;
    u32 gbuffer1_index;
    u32 gbuffer2_index;
    u32 depth_index;
    u32 lit_hdr_uav_index{0}; // For the compute version, just 0 always
    u32 sampler_index;
    const u32 shadow_texture_index;
    const u32 shadow_sampler_index;
    u32 debug_mode;
};
*/
struct DeferredLightingPushConstants {
    DeviceReadPtr<UBO> frame_ubo;
    DeviceReadPtr<PointLight> point_lights;
    DeviceReadPtr<Cluster> clusters; // Cluster offset/count data
    DeviceReadPtr<uint> cluster_light_indices; // Packed light indices
    float4x4 shadow_matrix;
    float log_z_scale;
    float z_near;
    float z_far;
    uint tiles_x;
    uint tiles_y;
    uint tiles_z;
    uint gbuffer0_index;
    uint gbuffer1_index;
    uint gbuffer2_index;
    uint ssao_index;
    uint depth_index;
    uint sampler_index;
    uint shadow_texture_index;
    uint shadow_sampler_index;
    uint debug_mode;
};

float linearize_reverse_z(float d, float z_near, float z_far) {
    // d = 1.0 at z_near, d = 0.0 at z_far
    return (z_near * z_far) / (z_near + d * (z_far - z_near));
}

uint compute_cluster_index(float2 screen_pos,
                           float raw_depth, // Sampled from Depth Texture (1.0 = near, 0.0 = far)
                           float z_near, float z_far, float log_z_scale, uint3 tiles, float2 screen_size) {
    uint cluster_x = uint(floor(screen_pos.x / screen_size.x * float(tiles.x)));
    uint cluster_y = uint(floor(screen_pos.y / screen_size.y * float(tiles.y)));

    float view_dist = linearize_reverse_z(raw_depth, z_near, z_far);

    uint cluster_z = 0;
    if (view_dist > z_near) {
        cluster_z = uint(log2(view_dist / z_near) * log_z_scale);
    }

    cluster_x = min(cluster_x, tiles.x - 1);
    cluster_y = min(cluster_y, tiles.y - 1);
    cluster_z = min(cluster_z, tiles.z - 1);

    return cluster_x + (cluster_y * tiles.x) + (cluster_z * tiles.x * tiles.y);
}

struct VSOut {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("fragment")] float4 fs_fullscreen_main(VSOut i, uniform DeferredLightingPushConstants pc) : SV_Target0 {
    float2 uv = i.uv;

    // Sample G-Buffer
    float4 g0 = sample_texture(uv, pc.gbuffer0_index, pc.sampler_index);
    float4 g1 = sample_texture(uv, pc.gbuffer1_index, pc.sampler_index);
    float4 g2 = sample_texture(uv, pc.gbuffer2_index, pc.sampler_index);
    float depth = sample_texture(uv, pc.depth_index).r;

    // Unpack G-Buffer
    float3 albedo = g0.rgb;
    float ao = saturate(g0.a);

    float rough = clamp(g1.g, 0.04, 1.0);
    float metal = saturate(g1.a);

    float3 emissive = g2.rgb;

    float2 oct_u = g1.xy;
    float3 N = oct_decode(oct_u);
    float3 world_pos = reconstruct_world_position_from_depth(uv, depth, pc.frame_ubo);
    float shadow_factor =
            calculate_shadow(world_pos, pc.shadow_matrix, pc.shadow_texture_index, pc.shadow_sampler_index);
    float3 V = safe_normalize(pc.frame_ubo->camera_position.xyz - world_pos);

    float ssao = sample_texture_lod(uv, pc.ssao_index, 0, pc.sampler_index).r;
    float combined_ao = ao * ssao;

    // Ambient
    float3 color = calculate_ambient(albedo, metal, combined_ao, float3(0.03, 0.03, 0.03), 3);

    // Sun light
    color += evaluate_directional_light(pc.frame_ubo->sun_dir, pc.frame_ubo->sun_intensity * shadow_factor,
                                        float3(1.0, 1.0, 1.0), V, N, albedo, rough, metal);

    float3 view_pos = mul(pc.frame_ubo->view, float4(world_pos, 1.0)).xyz;
    float2 screen_size = pc.frame_ubo->viewport_size;

    uint cluster_idx = compute_cluster_index(i.position.xy, depth, pc.z_near, pc.z_far, pc.log_z_scale,
                                             uint3(pc.tiles_x, pc.tiles_y, pc.tiles_z), pc.frame_ubo->viewport_size);

    var cluster = pc.clusters[cluster_idx];
    var count = min(cluster.light_count, MAX_LIGHTS_PER_CLUSTER);
    for (uint li = 0; li < count; ++li) {
        uint index_in_packed_buffer = cluster.light_offset + li;
        uint light_idx = pc.cluster_light_indices[index_in_packed_buffer];
        PointLight light = pc.point_lights[light_idx];
        color += evaluate_point_light_pbr(light, world_pos, V, N, albedo, rough, metal);
    }

    color += emissive;
    if (pc.debug_mode == 1) return float4(N * 0.5 + 0.5, 1.0);  // world normals
if (pc.debug_mode == 2) return float4(albedo, 1.0);           // albedo
if (pc.debug_mode == 3) return float4(float3(ao), 1.0);       // AO
if (pc.debug_mode == 4) return float4(float3(rough), 1.0);    // roughness
if (pc.debug_mode == 5) {
    uint id = sample_texture_uint(uv, pc.gbuffer2_index, 3, pc.sampler_index); // material ID stored in emissive alpha
    return float4(
        frac(float(id) * 0.1234),
        frac(float(id) * 0.5678),
        frac(float(id) * 0.9012),
        1.0
    );
}
    return float4(color, 1.0);
}
