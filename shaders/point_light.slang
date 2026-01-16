// light_visualization.slang

import common;

struct PushConstants
{
    UBO*             frame_ubo;
    PointLight*      compact;
    float3*          vertices;
    uint*            indices;
    IndirectCommand* indirect;
};

[[vk::push_constant]]
PushConstants g_push;

struct VertexOutput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
    float3 world_pos : TEXCOORD0;
};

// -----------------------------------------------------------------------------
// Vertex Shader with Programmable Vertex Pulling
// -----------------------------------------------------------------------------
[shader("vertex")]
VertexOutput main_vs(uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID)
{
    VertexOutput output;

    // Manually fetch index and vertex position
    uint index = g_push.indices[vertex_id];
    float3 local_pos = g_push.vertices[index];

    // Fetch the compacted light for this instance
    PointLight light = g_push.compact[instance_id];

    float3 light_world_pos = light.position_radius.xyz;
    float light_radius = light.position_radius.w;
    float3 light_color = light.colour_intensity.xyz;

    // Scale the unit cube vertex by the light radius and translate to light position
    float3 world_pos = local_pos * light_radius + light_world_pos;

    // Transform to clip space
float4 view_pos = mul(g_push.frame_ubo->view, float4(world_pos, 1.0));
float4 clip_pos = mul(g_push.frame_ubo->projection, view_pos);
    output.position = clip_pos;
    output.color = light_color;
    output.world_pos = world_pos;

    return output;
}

// -----------------------------------------------------------------------------
// Fragment Shader
// -----------------------------------------------------------------------------
[shader("fragment")]
float4 main_fs(VertexOutput input) : SV_Target0
{
    // Simple colored output with slight depth-based darkening for depth perception
    float3 color = input.color;

    // Optional: add some basic shading based on world position
    float3 light_dir = normalize(float3(1.0, 1.0, -1.0));
    float3 normal = normalize(cross(ddx(input.world_pos), ddy(input.world_pos)));
    float ndotl = max(0.2, dot(normal, light_dir));

    color *= ndotl;

    return float4(color, 1.0);
}

[shader("fragment")]
float4 debug_fs(VertexOutput input) : SV_Target0
{
    // 1. Output a very low intensity base value.
    // If 100 lights overlap, this will sum to 1.0 (pure red/white).
    // Adjust the 0.01 to change the "sensitivity" of the heatmap.
    float intensity = 0.01;

    // 2. Simple color ramp: Blue (low) -> Green -> Red (high)
    // Because we are using Additive Blending, we just return a low-intensity
    // vector that builds up.

    // To see individual light boundaries, we can use the world position
    // to add a subtle edge effect.
    float3 normal = normalize(cross(ddx(input.world_pos), ddy(input.world_pos)));
    float edge = 1.0 - max(0.0, dot(normal, float3(0, 0, 1))); // View-space normal check

    // Returning a very dim blue/green. As they add up, they will saturate.
    // Red channel grows fastest to show "heat"
    return float4(intensity * 2.0, intensity * 0.5, intensity * 0.1, 1.0);
}

[shader("fragment")]
float4 heat_fs(VertexOutput input) : SV_Target0
{
    // Adjust this 'alpha' to control sensitivity.
    // 0.05 means 20 lights reaching the same pixel will hit 'Full Heat'.
    float alpha = 0.05;

    // We output a color where the Red channel tracks the density.
    // In Additive Blending (One, One), these channels just stack up.
    return float4(alpha, 0.0, 0.0, 1.0);
}