import common;
import maths;
import pointer;

struct HeatmapPushConstants {
    DeviceReadPtr<Cluster> clusters;
    uint tiles_x;
    uint tiles_y;
    uint tiles_z;
    uint max_lights_per_cluster;
    uint debug_texture_uav_index;
    uint cell_size;
    uint slices_per_row;
};

float3 heat_colour(float t) {
    float3 col = float3(0.0);
    col = lerp(col,    float3(0.0, 0.0, 1.0), saturate(t * 5.0));
    col = lerp(col,    float3(0.0, 1.0, 1.0), saturate(t * 5.0 - 1.0));
    col = lerp(col,    float3(0.0, 1.0, 0.0), saturate(t * 5.0 - 2.0));
    col = lerp(col,    float3(1.0, 1.0, 0.0), saturate(t * 5.0 - 3.0));
    col = lerp(col,    float3(1.0, 0.0, 0.0), saturate(t * 5.0 - 4.0));
    return col;
}

[numthreads(8, 8, 1)]
[shader("compute")]
void ClusterHeatmapCS(
    uint3 dtid : SV_DispatchThreadID,
    uniform HeatmapPushConstants pc)
{
    uint px = dtid.x;
    uint py = dtid.y;

    uint total_width  = pc.tiles_x * pc.slices_per_row * pc.cell_size;
    uint slice_rows   = (pc.tiles_z + pc.slices_per_row - 1) / pc.slices_per_row;
    uint total_height = pc.tiles_y * slice_rows * pc.cell_size;

    if (px >= total_width || py >= total_height)
        return;

    // Which cell within the grid
    uint cell_x = px / pc.cell_size;
    uint cell_y = py / pc.cell_size;

    // Which slice column/row block
    uint slice_col = cell_x / pc.tiles_x;
    uint slice_row = cell_y / pc.tiles_y;
    uint cz = slice_row * pc.slices_per_row + slice_col;

    // Cluster XY within that slice
    uint cx = cell_x % pc.tiles_x;
    uint cy = cell_y % pc.tiles_y;

    if (cx >= pc.tiles_x || cy >= pc.tiles_y || cz >= pc.tiles_z)
        return;

    uint cluster_idx = cz * (pc.tiles_x * pc.tiles_y) + cy * pc.tiles_x + cx;
    Cluster cluster = pc.clusters[cluster_idx];

    float heat = saturate(float(cluster.light_count) / float(pc.max_lights_per_cluster));
    float3 colour = heat_colour(heat);

    RWTexture2D<float4> out_tex = get_storage_image(pc.debug_texture_uav_index);
    out_tex[uint2(px, py)] = float4(colour, 1.0);
}