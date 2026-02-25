import common;

bool sphere_in_frustum(float3 view_pos, float radius, float4 frustum_planes[6])
{
    for (int i = 0; i < 6; ++i)
    {
        if (dot(frustum_planes[i].xyz, view_pos) + frustum_planes[i].w < -radius)
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


float3 safe_normalize(float3 v)
{
    float l2 = dot(v, v);
    if (l2 <= 1e-20) return float3(0.0, 1.0, 0.0);
    return v * rsqrt(l2);
}


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

float3 unproject_to_view(float4x4 inv_proj, float2 ndc_xy, float linear_depth)
{
    // For a perspective projection, the view-space Z is -linear_depth (RH, looking down -Z).
    // Project that back to NDC Z using the projection terms directly, then unproject.
    // This avoids the fragile xy/z rescaling entirely.

    // We want view-space point (x, y, -linear_depth).
    // Unproject NDC xy at w=1 to get the ray direction at z=-1 in view space.
    float4 clip = float4(ndc_xy, 0.0, 1.0);
    float4 view = mul(inv_proj, clip);
    view /= view.w;

    // view.xyz is now a view-space point on the near plane.
    // Scale so that z = -linear_depth.
    float scale = -linear_depth / view.z;
    return view.xyz * scale;
}

struct ClusterViewData {
    uint tiles_x;
    uint tiles_y;
    uint tiles_z;
    float log_z_scale;
    float z_near;
    float z_far;
    float4x4 inv_projection;
    float4x4 view;
}

struct ViewAABB {
    float3 min_v;
    float3 max_v;
}

ViewAABB get_cluster_view_aabb(uint x, uint y, uint z, ClusterViewData pc) {
    // Map tile indices to NDC [-1, 1]
    // Vulkan Y is down: tile 0 is top (NDC -1), tile pc.tiles_y is bottom (NDC 1)
    float2 ndc_min = float2(
        (float(x) / pc.tiles_x) * 2.0 - 1.0,
        (float(y) / pc.tiles_y) * 2.0 - 1.0
    );
    float2 ndc_max = float2(
        (float(x + 1) / pc.tiles_x) * 2.0 - 1.0,
        (float(y + 1) / pc.tiles_y) * 2.0 - 1.0
    );

    // Calculate view-space distances (positive) using your exponential scale
float z_near_dist = pc.z_near * pow(2.0, float(z)     / pc.log_z_scale);
float z_far_dist  = pc.z_near * pow(2.0, float(z + 1) / pc.log_z_scale);

    // Get the 4 corners of the cluster frustum at near and far planes
    float3 p[8];
    p[0] = unproject_to_view(pc.inv_projection, float2(ndc_min.x, ndc_min.y), z_near_dist);
    p[1] = unproject_to_view(pc.inv_projection, float2(ndc_max.x, ndc_min.y), z_near_dist);
    p[2] = unproject_to_view(pc.inv_projection, float2(ndc_min.x, ndc_max.y), z_near_dist);
    p[3] = unproject_to_view(pc.inv_projection, float2(ndc_max.x, ndc_max.y), z_near_dist);
    p[4] = unproject_to_view(pc.inv_projection, float2(ndc_min.x, ndc_min.y), z_far_dist);
    p[5] = unproject_to_view(pc.inv_projection, float2(ndc_max.x, ndc_min.y), z_far_dist);
    p[6] = unproject_to_view(pc.inv_projection, float2(ndc_min.x, ndc_max.y), z_far_dist);
    p[7] = unproject_to_view(pc.inv_projection, float2(ndc_max.x, ndc_max.y), z_far_dist);

    ViewAABB aabb;
    aabb.min_v = aabb.max_v = p[0];
    [unroll]
    for(int i = 1; i < 8; ++i) {
        aabb.min_v = min(aabb.min_v, p[i]);
        aabb.max_v = max(aabb.max_v, p[i]);
    }
    return aabb;
}