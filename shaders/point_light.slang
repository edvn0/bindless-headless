import common;
import maths;

struct PushConstants
{
    UBO*             frame_ubo;
    PointLight*      compact;
    float3*          vertices;
    uint*            indices;
    IndirectCommand* indirect;
    IndirectMeshCommand* indirect_mesh;
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

struct TaskData {
    uint base_light_idx;
    uint lod_mask; // Bitmask: 2 bits per light to define LOD (0=Cull, 1=Point, 2=Quad, 3=Cube)
};

groupshared TaskData pl;

[shader("amplification")]
[numthreads(16, 1, 1)]
void main_ts(uint gtid : SV_GroupThreadID, uint gid : SV_GroupID) {
    pl.base_light_idx = gid * 16;
    pl.lod_mask = 0;

    uint light_idx = pl.base_light_idx + gtid;
    uint lod = 0; // 0: Cull, 1: Point, 2: Quad, 3: Cube

    if (light_idx < g_push.indirect->instance_count) {
        PointLight light = g_push.compact[light_idx];
        float3 center = light.position_radius.xyz;
        float radius = light.position_radius.w;

        // 1. Frustum Check
        if (sphere_in_frustum(center, radius, g_push.frame_ubo->frustum_planes)) {
            // 2. Screen Space Size Check
            float4 view_pos = mul(g_push.frame_ubo->view, float4(center, 1.0));
            // Project radius: approx pixels = (radius * screen_height) / (view_z * tan(fov/2))
            // Simpler: Use distance to determine LOD
           float distSq = dot(view_pos.xyz, view_pos.xyz);
           float relSizeSq = (radius * radius) / distSq;

           if (relSizeSq < 0.000004)      lod = 1; // Point
           else if (relSizeSq < 0.0001)   lod = 2; // Quad
           else                           lod = 3; // Cube
        }
    }

    // Pack LOD into the mask (2 bits per thread)
    uint group_lod_mask = WaveActiveBitOr(lod << (gtid * 2));

    if (gtid == 0) {
        pl.lod_mask = group_lod_mask;
        // Only dispatch if at least one light in the 16 is visible
        if (group_lod_mask != 0) {
            DispatchMesh(1, 1, 1, pl);
        } else {
            DispatchMesh(0, 0, 0, pl);
        }
    }
}

[shader("mesh")]
[numthreads(16, 1, 1)]
[outputtopology("triangle")]
void main_ms(
    uint gtid : SV_GroupThreadID,
    in payload TaskData payload,
    out vertices VertexOutput verts[128],
    out indices uint3 prims[192]
) {
    uint light_idx = payload.base_light_idx + gtid;
    // Extract 2-bit LOD: 0=None, 2=Quad, 3=Cube
    uint lod = (payload.lod_mask >> (gtid * 2)) & 0x3;

    // 1. Calculate Topology Needs
    uint vert_count = 0;
    uint tri_count  = 0;

    if (lod == 3) { // Cube
        vert_count = 8;
        tri_count  = 12;
    } else if (lod == 2) { // Quad (Billboard)
        vert_count = 4;
        tri_count  = 2;
    }

    // 2. Compact the Output Arrays
    // Every thread needs to know where its section starts
    uint v_base = WavePrefixSum(vert_count);
    uint t_base = WavePrefixSum(tri_count);

    // The last thread + its own count gives the total for the whole meshlet
    uint total_verts = WaveActiveSum(vert_count);
    uint total_tris  = WaveActiveSum(tri_count);

    SetMeshOutputCounts(total_verts, total_tris);

    if (lod == 0 || light_idx >= g_push.indirect->instance_count) return;

    PointLight light = g_push.compact[light_idx];
    float3 center = light.position_radius.xyz;
    float radius = light.position_radius.w;
    float3 color = light.colour_intensity.xyz;

    // 3. Geometry Generation
    if (lod == 3) {
        // --- CUBE LOD ---
        [unroll]
        for (uint i = 0; i < 8; i++) {
            float3 unit_pos = float3((i & 1) << 1, (i & 2), (i >> 1) & 2) - 1.0;
            float3 world_pos = unit_pos * radius + center;

            verts[v_base + i].position = mul(g_push.frame_ubo->projection,
                                         mul(g_push.frame_ubo->view, float4(world_pos, 1.0)));
            verts[v_base + i].color = color;
            verts[v_base + i].world_pos = world_pos;
        }

        // Indices for Cube
        uint3 cube_indices[12] = {
            uint3(0, 2, 1), uint3(1, 2, 3), uint3(4, 5, 6), uint3(5, 7, 6),
            uint3(0, 4, 2), uint3(2, 4, 6), uint3(1, 3, 5), uint3(3, 7, 5),
            uint3(0, 1, 4), uint3(1, 5, 4), uint3(2, 6, 3), uint3(3, 6, 7)
        };
        [unroll]
        for (uint j = 0; j < 12; j++) {
            prims[t_base + j] = cube_indices[j] + v_base;
        }
    }
    else if (lod == 2) {
        // --- QUAD LOD (Camera-Facing Billboard) ---
        // Extract right and up vectors from the view matrix (transposed/inverse)
        float3 right = float3(g_push.frame_ubo->view[0][0], g_push.frame_ubo->view[1][0], g_push.frame_ubo->view[2][0]);
        float3 up    = float3(g_push.frame_ubo->view[0][1], g_push.frame_ubo->view[1][1], g_push.frame_ubo->view[2][1]);

        float3 offsets[4] = {
            (-right - up) * radius, (right - up) * radius,
            (-right + up) * radius, (right + up) * radius
        };

        [unroll]
        for (uint i = 0; i < 4; i++) {
            float3 world_pos = center + offsets[i];
            verts[v_base + i].position = mul(g_push.frame_ubo->projection,
                                         mul(g_push.frame_ubo->view, float4(world_pos, 1.0)));
            verts[v_base + i].color = color;
            verts[v_base + i].world_pos = world_pos;
        }

        prims[t_base + 0] = uint3(v_base + 0, v_base + 1, v_base + 2);
        prims[t_base + 1] = uint3(v_base + 1, v_base + 3, v_base + 2);
    }
}