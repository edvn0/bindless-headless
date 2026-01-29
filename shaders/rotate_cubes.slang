// Match the structure used in your Vertex Shader
struct Transform {
    float3x4 m;
};

// Match your C++ RotateCubesPushConstant exactly
struct RotateCubesPushConstant {
    uint cube_count;
    float delta_time;
    float rads_per_second;
    Transform* transforms;
    Transform* previous_frame_transforms;
};

// Simple hash to replace your CPU random_factors array
float3 get_random_factor(uint index) {
    uint h = index * 1103515245u + 12345u;
    h ^= h >> 16;
    float3 factors;
    factors.x = float(h & 0xFF) / 255.0;
    factors.y = float((h >> 8) & 0xFF) / 255.0;
    factors.z = float((h >> 16) & 0xFF) / 255.0;
    return factors; 
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rotate_cs(uint3 threadID : SV_DispatchThreadID, uniform RotateCubesPushConstant pc)
{
    uint idx = threadID.x;
    if (idx >= pc.cube_count) return;
    
    // Load from previous frame's slot
    float3x4 m3x4 = pc.previous_frame_transforms[idx].m;
    
    // Reconstruct 4x4 to perform rotation math
    float4x4 model = float4x4(
        m3x4._m00_m01_m02_m03,
        m3x4._m10_m11_m12_m13,
        m3x4._m20_m21_m22_m23,
        float4(0, 0, 0, 1)
    ); 
    
    // Get the per-instance rotation multiplier
    float3 random_f = get_random_factor(idx);
    float3 angles = pc.rads_per_second * pc.delta_time * random_f;
    
    // Construct rotation matrices for X, Y, Z
    // This replicates the GLM order: Y * Z * X
    float3 s = sin(angles);
    float3 c = cos(angles);
    
    float4x4 rotX = float4x4(
        1, 0, 0, 0,
        0, c.x, -s.x, 0,
        0, s.x, c.x, 0,
        0, 0, 0, 1
    );
    
    float4x4 rotY = float4x4(
        c.y, 0, s.y, 0,
        0, 1, 0, 0,
        -s.y, 0, c.y, 0,
        0, 0, 0, 1
    );
    
    float4x4 rotZ = float4x4(
        c.z, -s.z, 0, 0,
        s.z, c.z, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    );
    
    // Combine rotations: rot = Y * Z * X
    float4x4 combinedRot = mul(rotY, mul(rotZ, rotX));
    
    // Apply rotation to current transform
    float4x4 result = mul(combinedRot, model);
    
    // Extract and preserve the scale from each axis
    float scale_x = length(result[0].xyz);
    float scale_y = length(result[1].xyz);
    float scale_z = length(result[2].xyz);
    
    // Orthonormalize the rotation part to prevent matrix degradation
    float3 x_axis = result[0].xyz / scale_x;
    float3 y_axis = result[1].xyz / scale_y;
    y_axis = y_axis - dot(y_axis, x_axis) * x_axis;
    y_axis = normalize(y_axis);
    float3 z_axis = cross(x_axis, y_axis);
    
    // Reapply the scale to the orthonormal axes
    result[0] = float4(x_axis * scale_x, result[0].w);
    result[1] = float4(y_axis * scale_y, result[1].w);
    result[2] = float4(z_axis * scale_z, result[2].w);
    
    // Write back to current frame's slot
    pc.transforms[idx].m = float3x4(
        result[0],
        result[1],
        result[2]
    );
}
