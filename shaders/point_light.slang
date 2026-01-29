import common;
import maths;

// -----------------------------------------------------------------------------
// PBR Helper Functions
// -----------------------------------------------------------------------------

static const float PI = 3.14159265359;

// Schlick's approximation for Fresnel
float3 fresnel_schlick(float cos_theta, float3 f0)
{
    return f0 + (1.0 - f0) * pow(saturate(1.0 - cos_theta), 5.0);
}

// GGX/Trowbridge-Reitz normal distribution function
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

// Schlick-GGX geometry function
float geometry_schlick_ggx(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    
    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    
    return num / max(denom, 0.0001);
}

// Smith's method for geometry obstruction
float geometry_smith(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = saturate(dot(N, V));
    float NdotL = saturate(dot(N, L));
    float ggx2 = geometry_schlick_ggx(NdotV, roughness);
    float ggx1 = geometry_schlick_ggx(NdotL, roughness);
    
    return ggx1 * ggx2;
}

// -----------------------------------------------------------------------------
// Material Sampling
// -----------------------------------------------------------------------------

struct SurfaceMaterial
{
    float3 albedo;
    float3 normal;
    float  roughness;
    float  metallic;
    float  occlusion;
    float3 emissive;
};

// Material flag constants (must match GPUMaterialData)
static const uint FLAG_ALBEDO_MAP    = 1 << 0;
static const uint FLAG_NORMAL_MAP    = 1 << 1;
static const uint FLAG_ROUGHNESS_MAP = 1 << 2;
static const uint FLAG_METALLIC_MAP  = 1 << 3;
static const uint FLAG_OCCLUSION_MAP = 1 << 4;
static const uint FLAG_EMISSIVE_MAP  = 1 << 5;

// Sample and decode normal map (tangent space -> world space)
float3 sample_normal_map(
    float2 uv,
    uint normal_map_index,
    bool has_map,
    float3 world_normal,
    float3 world_pos,
    uint sampler_index = 0)
{
    if (!has_map) {
        return normalize(world_normal);
    }
    
    // Sample normal map
    float3 tangent_normal = sample_texture(uv, normal_map_index, sampler_index).xyz;
    tangent_normal = tangent_normal * 2.0 - 1.0;
    
    // Compute tangent basis from world normal and derivatives
    float3 Q1 = ddx(world_pos);
    float3 Q2 = ddy(world_pos);
    float2 st1 = ddx(uv);
    float2 st2 = ddy(uv);
    
    float3 N = normalize(world_normal);
    float3 T = normalize(Q1 * st2.y - Q2 * st1.y);
    float3 B = -normalize(cross(N, T));
    float3x3 TBN = float3x3(T, B, N);
    
    return normalize(mul(tangent_normal, TBN));
}

// Evaluate material from textures and factors
SurfaceMaterial evaluate_material(
    Material mat,
    float2 uv,
    float3 world_normal,
    float3 world_pos)
{
    SurfaceMaterial surf;
    
    // Albedo
    if ((mat.flags & FLAG_ALBEDO_MAP) != 0) {
        float4 albedo_sample = sample_texture(uv, mat.albedo_map, 0);
        surf.albedo = albedo_sample.rgb * mat.albedo_factor.rgb;
    } else {
        surf.albedo = mat.albedo_factor.rgb;
    }
    
    // Normal
    surf.normal = sample_normal_map(
        uv,
        mat.normal_map,
        (mat.flags & FLAG_NORMAL_MAP) != 0,
        world_normal,
        world_pos,
        0
    );
    
    // Roughness
    if ((mat.flags & FLAG_ROUGHNESS_MAP) != 0) {
        float roughness_sample = sample_texture(uv, mat.roughness_map, 0).r;
        surf.roughness = roughness_sample * mat.roughness_factor;
    } else {
        surf.roughness = mat.roughness_factor;
    }
    surf.roughness = clamp(surf.roughness, 0.04, 1.0);
    
    // Metallic
    if ((mat.flags & FLAG_METALLIC_MAP) != 0) {
        float metallic_sample = sample_texture(uv, mat.metallic_map, 0).r;
        surf.metallic = metallic_sample * mat.metallic_factor;
    } else {
        surf.metallic = mat.metallic_factor;
    }
    surf.metallic = saturate(surf.metallic);
    
    // Occlusion
    if ((mat.flags & FLAG_OCCLUSION_MAP) != 0) {
        surf.occlusion = sample_texture(uv, mat.occlusion_map, 0).r;
    } else {
        surf.occlusion = 1.0;
    }
    
    // Emissive
    if ((mat.flags & FLAG_EMISSIVE_MAP) != 0) {
        float3 emissive_sample = sample_texture(uv, mat.emissive_map, 0).rgb;
        surf.emissive = emissive_sample * mat.emissive_factor;
    } else {
        surf.emissive = mat.emissive_factor;
    }
    
    return surf;
}

// -----------------------------------------------------------------------------
// Lighting Calculations
// -----------------------------------------------------------------------------

// Cook-Torrance BRDF for a single light
float3 calculate_direct_lighting(
    float3 L,              // Light direction (to light)
    float3 V,              // View direction (to camera)
    float3 N,              // Surface normal
    float3 albedo,
    float  roughness,
    float  metallic,
    float3 radiance)       // Incoming light color * intensity
{
    float3 H = normalize(V + L);
    
    // Calculate F0 (surface reflection at zero incidence)
    float3 f0 = float3(0.04, 0.04, 0.04);
    f0 = lerp(f0, albedo, metallic);
    
    // Cook-Torrance BRDF
    float NDF = distribution_ggx(N, H, roughness);
    float G = geometry_smith(N, V, L, roughness);
    float3 F = fresnel_schlick(saturate(dot(H, V)), f0);
    
    float3 numerator = NDF * G * F;
    float denominator = 4.0 * saturate(dot(N, V)) * saturate(dot(N, L));
    float3 specular = numerator / max(denominator, 0.001);
    
    // Energy conservation
    float3 kS = F;
    float3 kD = float3(1.0, 1.0, 1.0) - kS;
    kD *= 1.0 - metallic;
    
    float NdotL = saturate(dot(N, L));
    
    return (kD * albedo / PI + specular) * radiance * NdotL;
}

// Directional/sun light
float3 evaluate_directional_light(
    float3 light_dir,      // Direction light is coming FROM
    float  light_intensity,
    float3 light_color,
    float3 V,
    float3 N,
    float3 albedo,
    float  roughness,
    float  metallic)
{
    float3 L = -light_dir;  // Direction TO light
    float3 radiance = light_color * light_intensity;
    
    return calculate_direct_lighting(L, V, N, albedo, roughness, metallic, radiance);
}

// Point light with distance attenuation
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
    float  light_radius = light.position_radius.w;
    float3 light_color = light.colour_intensity.rgb;
    float  light_intensity = light.colour_intensity.a;
    
    float3 L = light_pos - world_pos;
    float distance = length(L);
    L = L / distance;  // Normalize
    
    // Distance attenuation (inverse square with radius cutoff)
    float attenuation = 1.0 / (distance * distance);
    
    // Smooth falloff near radius
    float falloff = 1.0 - smoothstep(light_radius * 0.75, light_radius, distance);
    attenuation *= falloff;
    
    float3 radiance = light_color * light_intensity * attenuation;
    
    return calculate_direct_lighting(L, V, N, albedo, roughness, metallic, radiance);
}

// Simple ambient/IBL approximation
float3 calculate_ambient(
    float3 albedo,
    float  metallic,
    float  occlusion,
    float3 ambient_color,
    float  ambient_intensity)
{
    // Very simple ambient - could be replaced with actual IBL
    float3 f0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);
    float3 kD = (1.0 - f0) * (1.0 - metallic);
    
    return kD * albedo * ambient_color * ambient_intensity * occlusion;
}

// -----------------------------------------------------------------------------
// Full PBR Lighting Function
// -----------------------------------------------------------------------------

struct LightingParams
{
    UBO* frame_ubo;
    PointLight* point_lights;
    uint point_light_count;
    float3 ambient_color;
    float  ambient_intensity;
};

float3 calculate_pbr_lighting(
    SurfaceMaterial surf,
    float3 world_pos,
    float3 camera_pos,
    LightingParams lighting)
{
    float3 V = normalize(camera_pos - world_pos);
    float3 N = surf.normal;
    
    // Start with ambient
    float3 color = calculate_ambient(
        surf.albedo,
        surf.metallic,
        surf.occlusion,
        lighting.ambient_color,
        lighting.ambient_intensity
    );
    
    // Add directional/sun light
    color += evaluate_directional_light(
        lighting.frame_ubo->sun_dir,
        lighting.frame_ubo->sun_intensity,
        float3(1.0, 1.0, 1.0),  // White sun
        V, N,
        surf.albedo,
        surf.roughness,
        surf.metallic
    );
    
    // Add point lights
    for (uint i = 0; i < lighting.point_light_count; ++i)
    {
        color += evaluate_point_light(
            lighting.point_lights[i],
            world_pos,
            V, N,
            surf.albedo,
            surf.roughness,
            surf.metallic
        );
    }
    
    // Add emissive
    color += surf.emissive;
    
    return color;
}

// -----------------------------------------------------------------------------
// Shader Variants
// -----------------------------------------------------------------------------

struct VertexOutput {
    float4 position  : SV_Position;
    float3 world_pos : TEXCOORD0;
    float3 normal    : TEXCOORD1;
    float2 uv        : TEXCOORD2;
    nointerpolation uint material : TEXCOORD3;
};

struct VertexInput {
    float3 position;
    uint normal;
    uint uvs;
};

struct PC {
    UBO* frame_ubo;
    Transform* transforms;
    uint* draw_material_ids;
    Material* materials;
    uint base_draw_id;
    PointLight* point_lights;
    uint* point_light_count;
    float ambient_intensity;
};

[shader("vertex")]
VertexOutput main_vs(
    VertexInput input,
    uint instance : SV_InstanceID,
    uniform PC pc)
{
    VertexOutput o = {};
    float3x4 model_3x4 = pc.transforms[instance].m;
    
    float4x4 model = float4x4(
        model_3x4._m00_m01_m02_m03,
        model_3x4._m10_m11_m12_m13,
        model_3x4._m20_m21_m22_m23,
        float4(0, 0, 0, 1)
    );
    
    float4 world_pos = mul(model, float4(input.position, 1.0));
    o.position = mul(pc.frame_ubo->view_projection, world_pos);
    o.world_pos = world_pos.xyz;
    
    // Normal
    float3 local_normal = normalize(unpackSnorm3x10_1x2(input.normal).xyz);
    o.normal = normalize(mul((float3x3)model, local_normal));
    
    // UVs
    UV4 uvpack = unpack_uv_8_8_8_8(input.uvs);
    float2 uv_lerp = saturate(input.position.xy * 0.5 + 0.5);
    o.uv = lerp(uvpack.uv0, uvpack.uv1, uv_lerp);
    
    return o;
}

[shader("vertex")]
VertexOutput main_vs_mdi(
    VertexInput input,
    uint draw_id : SV_DrawIndex,
    uint instance_id : SV_InstanceID,
    uint base_instance : SV_StartInstanceLocation,
    uniform PC pc)
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
    o.position = mul(pc.frame_ubo->view_projection, world_pos);
    o.world_pos = world_pos.xyz;
    
    float3 local_normal = normalize(unpackSnorm3x10_1x2(input.normal).xyz);
    o.normal = normalize(mul((float3x3)model, local_normal));
    
    UV4 uvpack = unpack_uv_8_8_8_8(input.uvs);
    float2 uv_lerp = saturate(input.position.xy * 0.5 + 0.5);
    o.uv = lerp(uvpack.uv0, uvpack.uv1, uv_lerp);
    
    o.material = material_id;
    
    return o;
}

// -----------------------------------------------------------------------------
// Fragment Shader - Full PBR
// -----------------------------------------------------------------------------

[shader("fragment")]
float4 main_fs(VertexOutput input, uniform PC pc) : SV_Target0
{
    // Evaluate material
    Material mat = pc.materials[input.material];
    SurfaceMaterial surf = evaluate_material(
        mat,
        input.uv,
        input.normal,
        input.world_pos
    );
    
    // Setup lighting parameters
    LightingParams lighting;
    lighting.frame_ubo = pc.frame_ubo;
    lighting.point_lights = pc.point_lights;
    lighting.point_light_count = *pc.point_light_count;
    lighting.ambient_color = float3(0.03, 0.03, 0.03);
    lighting.ambient_intensity = pc.ambient_intensity;
    
    // Calculate lighting
    float3 color = calculate_pbr_lighting(
        surf,
        input.world_pos,
        pc.frame_ubo->camera_position.xyz,
        lighting
    );
    
    // Simple tone mapping
    color = color / (color + float3(1.0, 1.0, 1.0));
    
    // Gamma correction
    color = pow(color, float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
    
    return float4(color, 1.0);
}

// -----------------------------------------------------------------------------
// Simpler variant for testing/debugging
// -----------------------------------------------------------------------------

[shader("fragment")]
float4 main_fs_simple(VertexOutput input, uniform PC pc) : SV_Target0
{
    Material mat = pc.materials[input.material];
    SurfaceMaterial surf = evaluate_material(
        mat,
        input.uv,
        input.normal,
        input.world_pos
    );
    
    float3 V = normalize(pc.frame_ubo->camera_position.xyz - input.world_pos);
    float3 N = surf.normal;
    float3 L = -pc.frame_ubo->sun_dir;
    
    // Simple lighting
    float NdotL = saturate(dot(N, L));
    float3 ambient = surf.albedo * 0.03 * surf.occlusion;
    float3 diffuse = surf.albedo * NdotL * pc.frame_ubo->sun_intensity;
    
    float3 color = ambient + diffuse + surf.emissive;
    
    return float4(color, 1.0);
}
