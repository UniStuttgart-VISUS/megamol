/**
 * MegaMol
 * Copyright (c) 2024, MegaMol Dev Team
 * All rights reserved.
 */

#version 450

uniform sampler2D color_point_tx2D;
uniform sampler2D color_linear_tx2D;
uniform sampler2D coc_point_tx2D;

layout(binding = 0, r11f_g11f_b10f) writeonly uniform image2D color_4_tx2D;
layout(binding = 1, r11f_g11f_b10f) writeonly uniform image2D color_mul_coc_far_4_tx2D;
layout(binding = 2, rg8) writeonly uniform image2D coc_4_tx2D;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    uvec3 gID = gl_GlobalInvocationID.xyz;
    ivec2 pixel_coords = ivec2(gID.xy);
    ivec2 tgt_res = imageSize(color_4_tx2D);

    if (pixel_coords.x >= tgt_res.x || pixel_coords.y >= tgt_res.y) {
        return;
    }

    vec2 pixel_coord_00 = pixel_coords + vec2(-0.5, -0.5); // TODO: factor depending on pixel_coords size, TODO: the offset fucks up for outer perimeter pixels
    vec2 pixel_coord_10 = pixel_coords + vec2( 0.5, -0.5); // TODO: factor depending on pixel_coords size
    vec2 pixel_coord_01 = pixel_coords + vec2(-0.5,  0.5); // TODO: factor depending on pixel_coords size
    vec2 pixel_coord_11 = pixel_coords + vec2( 0.5,  0.5); // TODO: factor depending on pixel_coords size

    vec3 color = texelFetch(color_linear_tx2D, pixel_coords, 0);
    vec2 coc = texelFetch(coc_point_tx2D, pixel_coord_01, 0); // TODO: does this need to be pixel_coord_01 (top left) because of coordinate system differences between opengl/directx

    float coc_far_00 = texelFetch(coc_point_tx2D, pixel_coord_00, 0).y;
    float coc_far_10 = texelFetch(coc_point_tx2D, pixel_coord_10, 0).y;
    float coc_far_01 = texelFetch(coc_point_tx2D, pixel_coord_01, 0).y;
    float coc_far_11 = texelFetch(coc_point_tx2D, pixel_coord_11, 0).y;

    float weight_00 = 1000.0;
    vec3 color_mul_coc_far = weight_00 * texelFetch(color_point_tx2D, pixel_coord_00, 0); // TODO: is vec3 correct?
    float weight_sum = weight_00;

    float weight_10 = 1.0 / (abs(coc_far_00 - coc_far_10) + 0.001);
    color_mul_coc_far += weight_10 * texelFetch(color_point_tx2D, pixel_coord_10, 0);
    weight_sum += weight_00;

    float weight_01 = 1.0 / (abs(coc_far_00 - coc_far_01) + 0.001);
    color_mul_coc_far += weight_01 * texelFetch(color_point_tx2D, pixel_coord_01, 0);
    weight_sum += weight_01;

    float weight_11 = 1.0 / (abs(coc_far_00 - coc_far_11) + 0.001);
    color_mul_coc_far += weight_11 * texelFetch(color_point_tx2D, pixel_coord_11, 0);
    weight_sum += weight_11;

    color_mul_coc_far /= weight_sum;
    color_mul_coc_far *= coc.y;


    imageStore(color_4_tx2D, pixel_coords, color);
    imageStore(coc_4_tx2D, pixel_coords, coc);
    imageStore(color_mul_coc_far_4_tx2D, pixel_coords, color_mul_coc_far);
}
