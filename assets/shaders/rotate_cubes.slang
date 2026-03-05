import common;
import pointer;

struct RotatePushConstant {
    float delta_time;
    float rads_per_second;
    float total_time;
    uint count; // cube_count for geometry, light_count for lights
    DeviceReadPtr<InstanceData> previous_frame_transforms; // unused in light kernel
    DeviceWritePtr<InstanceData> transforms;
    DeviceReadPtr<PointLight> previous_point_lights;
    DeviceWritePtr<PointLight> point_lights;
    DeviceReadPtr<PointLight> static_point_lights;
};

float3 get_random_factor(uint index) {
    uint h = index * 1103515245u + 12345u;
    h ^= h >> 16;
    float3 factors;
    factors.x = float(h & 0xFF) / 255.0;
    factors.y = float((h >> 8) & 0xFF) / 255.0;
    factors.z = float((h >> 16) & 0xFF) / 255.0;
    return factors;
}

float3 get_light_offset_dir(uint index) {
    float phase = float(index % 3);
    if (phase == 0.0) return float3(1, 0, 0); // Move on X
    if (phase == 1.0) return float3(0, 0, 1); // Move on Z
    return float3(0, 1, 0);                   // Move on Y
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rotate_geometry_cs(uint3 tid : SV_DispatchThreadID, uniform RotatePushConstant pc)
{
   /* uint idx = tid.x;
    if (idx >= pc.count) return;

    float3x4 m3x4 = pc.previous_frame_transforms[idx].m;
    float4x4 model = float4x4(
        m3x4._m00_m01_m02_m03,
        m3x4._m10_m11_m12_m13,
        m3x4._m20_m21_m22_m23,
        float4(0, 0, 0, 1)
    );

    float3 random_f = get_random_factor(idx);
    float3 angles = pc.rads_per_second * pc.delta_time * random_f;
    float3 s = sin(angles);
    float3 c = cos(angles);

    float4x4 rot_x = float4x4(1,0,0,0,  0,c.x,-s.x,0,  0,s.x,c.x,0,  0,0,0,1);
    float4x4 rot_y = float4x4(c.y,0,s.y,0,  0,1,0,0,  -s.y,0,c.y,0,  0,0,0,1);
    float4x4 rot_z = float4x4(c.z,-s.z,0,0,  s.z,c.z,0,0,  0,0,1,0,  0,0,0,1);

    float4x4 result = mul(rot_y, mul(rot_z, mul(rot_x, model)));

    // Orthonormalize to prevent drift
    float scale_x = length(result[0].xyz);
    float scale_y = length(result[1].xyz);
    float scale_z = length(result[2].xyz);

    float3 x_axis = result[0].xyz / scale_x;
    float3 y_axis = normalize(result[1].xyz / scale_y - dot(result[1].xyz / scale_y, x_axis) * x_axis);
    float3 z_axis = cross(x_axis, y_axis);

    result[0] = float4(x_axis * scale_x, result[0].w);
    result[1] = float4(y_axis * scale_y, result[1].w);
    result[2] = float4(z_axis * scale_z, result[2].w);

    pc.transforms[idx].m = float3x4(result[0], result[1], result[2]);
    */
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rotate_lights_cs(uint3 tid : SV_DispatchThreadID, uniform RotatePushConstant pc)
{
    uint idx = tid.x;
    if (idx >= pc.count) return;
    var l = pc.static_point_lights[idx];

    // 1. Get the original static position
    float4 base = l.position_radius;
    float3 center = base.xyz;

    // 2. Calculate the orbit
    // We can add a per-light phase offset (idx * 0.5) so they don't all move in a line
    float angle = pc.total_time + (float(idx) * 13.5f);
    
    float offset_x = cos(0.02f*angle) * l.radius;
    float offset_z = sin(0.02f*angle) * l.radius;

    // 3. Apply the offset to the x and z coordinates
    float3 orbiting_pos = center + float3(offset_x, 0.0f, offset_z);

    // 4. Update the dynamic buffer
    pc.point_lights[idx].position_radius = float4(orbiting_pos, base.w);
    pc.point_lights[idx].colour_intensity = l.colour_intensity;
}