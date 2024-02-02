/**
 * MegaMol
 * Copyright (c) 2024, MegaMol Dev Team
 * All rights reserved.
 */

#version 450

uniform sampler2D coc_4_tx2D;
uniform sampler2D coc_near_blurred_4_tx2D;
uniform sampler2D near_4_tx2D;
uniform sampler2D far_4_tx2D;

layout(binding = 0, r11f_g11f_b10f) writeonly uniform image2D near_fill_4_tx2D;
layout(binding = 1, r11f_g11f_b10f) writeonly uniform image2D far_fill_4_tx2D;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    uvec3 gID = gl_GlobalInvocationID.xyz;
    ivec2 pixel_coords = ivec2(gID.xy);
    ivec2 tgt_resolution = imageSize(tgt_tx2D);

    if (pixel_coords.x >= tgt_resolution.x || pixel_coords.y >= tgt_resolution.y) {
        return;
    }

    vec3 near_fill = vec3(0.0);
    float coc_near_blurred = texelFetch(coc_near_blurred_4_tx2D, pixel_coords, 0).x; // TODO: pointSampler
    if(coc_near_blurred > 0.0) {
        for(int i = -1; i <= 1; ++i) {
            for(int j = -1; j <= 1; ++j) {
                vec3 sample = texelFetch(near_4_tx2D, pixel_coords + ivec2(i, j), 0); // TODO: pointSampler
                near_fill = max(near_fill, sample);
            }
        }
    }

    vec3 far_fill = vec3(0.0);
    float coc_far = texelFetch(coc_4_tx2D, pixel_coords, 0).y; // TODO: pointSampler
    if(coc_far > 0.0) {
        for(int i = -1; i <= 1; ++i) {
            for(int j = -1; j <= 1; ++j) {
                vec3 sample = texelFetch(far_4_tx2D, pixel_coords + ivec2(i, j), 0); // TODO: pointSampler
                far_fill = max(far_fill, sample);
            }
        }
    }


    imageStore(near_fill_4_tx2D, pixel_coords, near_fill);
    imageStore(far_fill_4_tx2D, pixel_coords, far_fill);
}
