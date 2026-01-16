import common;

bool sphere_in_frustum(float3 view_pos, float radius, float4 frustum_planes[6])
{
    for (int i = 0; i < 6; ++i) {
        float3 normal = frustum_planes[i].xyz;
        float d = frustum_planes[i].w;

        float dist = dot(normal, view_pos) + d;

        if (dist < -radius)
            return false;
    }

    return true;
}

bool light_in_frustum(PointLight l, float4x4 view, float4 frustum_planes[6])
{
    float3 world_pos = l.position_radius.xyz;
    float  radius    = l.position_radius.w;

    float4 view_pos4 = mul(view, float4(world_pos, 1.0f));
    float3 view_pos = view_pos4.xyz;

    return sphere_in_frustum(view_pos, radius, frustum_planes);
}

bool aabb_in_frustum(float3 bmin, float3 bmax, float4 planes[6]) {
    for (int i = 0; i < 6; i++) {
        float3 p = bmin;
        if (planes[i].x >= 0) p.x = bmax.x;
        if (planes[i].y >= 0) p.y = bmax.y;
        if (planes[i].z >= 0) p.z = bmax.z;

        if (dot(planes[i].xyz, p) + planes[i].w < 0) return false;
    }
    return true;
}