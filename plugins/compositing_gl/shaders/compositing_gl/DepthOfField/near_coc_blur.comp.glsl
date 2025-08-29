/**
 * MegaMol
 * Copyright (c) 2024, MegaMol Dev Team
 * All rights reserved.
 */

#version 450

uniform sampler2D coc_4_point_tx2D;

layout(binding = 0, r8) writeonly uniform image2D coc_near_blurred_4_tx2D;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    uvec3 gID = gl_GlobalInvocationID.xyz;
    ivec2 pixel_coords = ivec2(gID.xy);
    ivec2 tgt_res = imageSize(coc_near_blurred_4_tx2D);
    vec2 tex_coords = (vec2(pixel_coords) + vec2(0.5)) / vec2(tgt_res);

#ifdef MAX_FILTER_HORIZONTAL
    // PASS 1: horizontal max filter

    float mx = -1.0; // 0.0 should be sufficient, since coc values get clamped in first pass to [0.0, 1.0]
    vec2 step = vec2(1.0 / tgt_res.x, 0);

    for(int i = -6; i <= 6; ++i) {
        vec2 offset_coords = tex_coords + i * step;
        mx = max(mx, textureLod(coc_4_point_tx2D, offset_coords, 0).x);
    }

    imageStore(coc_near_blurred_4_tx2D, pixel_coords, mx.xxxx);

#elif defined(MAX_FILTER_VERTICAL)
    // PASS 2: veritcal max filter

    float mx = -1.0;
    vec2 step = vec2(0, 1.0 / tgt_res.y);

    for(int i = -6; i <= 6; ++i) {
        vec2 offset_coords = tex_coords + i * step;
        mx = max(mx, textureLod(coc_4_point_tx2D, offset_coords, 0).x);
    }

    imageStore(coc_near_blurred_4_tx2D, pixel_coords, mx.xxxx);

#elif defined(BLUR_FILTER_HORIZONTAL)
    // PASS 3: horizontal blur filter

    float blur = 0.0;
    vec2 step = vec2(1.0 / tgt_res.x, 0);

    for(int i = -6; i <= 6; ++i) {
        vec2 offset_coords = tex_coords + i * step;
        blur += textureLod(coc_4_point_tx2D, offset_coords, 0).x;
    }

    blur /= 13.0;

    imageStore(coc_near_blurred_4_tx2D, pixel_coords, blur.xxxx);

#elif defined(BLUR_FILTER_VERTICAL)
    // PASS 4: vertical blur filter

    float blur = 0.0;
    vec2 step = vec2(0, 1.0 / tgt_res.y);

    for(int i = -6; i <= 6; ++i) {
        vec2 offset_coords = tex_coords + i * step;
        blur += textureLod(coc_4_point_tx2D, offset_coords, 0).x;
    }

    blur /= 13.0;

    imageStore(coc_near_blurred_4_tx2D, pixel_coords, blur.xxxx);

#endif

}
