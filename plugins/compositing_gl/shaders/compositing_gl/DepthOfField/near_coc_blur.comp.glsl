/**
 * MegaMol
 * Copyright (c) 2024, MegaMol Dev Team
 * All rights reserved.
 */

#version 450

uniform sampler2D coc_4_tx2D;

layout(binding = 0, r8ui) writeonly uniform image2D coc_near_blurred_4_tx2D;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    uvec3 gID = gl_GlobalInvocationID.xyz;
    ivec2 pixel_coords = ivec2(gID.xy);
    ivec2 tgt_resolution = imageSize(tgt_tx2D);

    if (pixel_coords.x >= tgt_resolution.x || pixel_coords.y >= tgt_resolution.y) {
        return;
    }

#ifdef MAX_FILTER_HORIZONTAL
    // PASS 1: horizontal max filter

    int mx = INT_MIN;
    ivec2 step = ivec2(-6, 0);

    for(int i = -6; i <= 6; ++i) {
        mx = max(mx, texture(coc_4_tx2D, pixel_coords + step).r);
        step.x++;
    }

    imageStore(coc_near_blurred_4_tx2D, pixel_coords, (uint)mx);

#elif MAX_FILTER_VERTICAL
    // PASS 2: veritcal max filter

    int mx = INT_MIN
    ivec2 step = ivec2(0, -6);

    for(int i = -6; i <= 6; ++i) {
        mx = max(mx, texture(coc_4_tx2D, pixel_coords + step).r);
        step.y++;
    }

    imageStore(coc_near_blurred_4_tx2D, pixel_coords, (uint)mx);

#elif BLUR_FILTER_HORIZONTAL
    // PASS 3: horizontal blur filter

    int blur = 0;
    ivec2 step = ivec2(0, 0);

    for(int i = 0; i <= 12; ++i) {
        blur += texture(coc_4_tx2D, pixel_coords + step).r;
        step.x++;
    }

    imageStore(coc_near_blurred_4_tx2D, pixel_coords, blur / 13);

#else
    // PASS 4: vertical blur filter

    int blur = 0;
    ivec2 step = ivec2(0, 0);

    for(int i = 0; i <= 12; ++i) {
        blur += texture(coc_4_tx2D, pixel_coords + step).r;
        step.y++;
    }

    imageStore(coc_near_blurred_4_tx2D, pixel_coords, blur / 13);

#endif

}
