static const float BAYER_4x4[16] = {
    0.0/16.0,  8.0/16.0,  2.0/16.0,  10.0/16.0,
    12.0/16.0, 4.0/16.0,  14.0/16.0, 6.0/16.0,
    3.0/16.0,  11.0/16.0, 1.0/16.0,  9.0/16.0,
    15.0/16.0, 7.0/16.0,  13.0/16.0, 5.0/16.0
};

bool is_dithered_discarded(float alpha, float2 screen_pos, uint frame_index = 0) {
    uint x = (uint(screen_pos.x) + (frame_index * 13)) % 4;
    uint y = (uint(screen_pos.y) + (frame_index * 7)) % 4;

    float threshold = BAYER_4x4[y * 4 + x];
    return alpha < threshold;
}
