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
    vec2 tex_coords = (vec2(pixel_coords) + vec2(0.5)) / vec2(tgt_res);

    vec2 quarter_shift = vec2(0.25) / vec2(tgt_res);

    vec2 tex_coord_00 = tex_coords + vec2(-quarter_shift.x, -quarter_shift.y);
    vec2 tex_coord_10 = tex_coords + vec2( quarter_shift.x, -quarter_shift.y);
    vec2 tex_coord_01 = tex_coords + vec2(-quarter_shift.x,  quarter_shift.y); // top left
    vec2 tex_coord_11 = tex_coords + vec2( quarter_shift.x,  quarter_shift.y);

    vec4 color = textureLod(color_linear_tx2D, tex_coords, 0);
    vec4 coc = textureLod(coc_point_tx2D, tex_coord_01, 0); // get top left


    // custom bilinear filtering
    // source: https://github.com/maxest/MaxestFramework/blob/master/samples/dof/data/downsample_ps.hlsl

    float coc_far_00 = textureLod(coc_point_tx2D, tex_coord_00, 0).y;
    float coc_far_10 = textureLod(coc_point_tx2D, tex_coord_10, 0).y;
    float coc_far_01 = textureLod(coc_point_tx2D, tex_coord_01, 0).y;
    float coc_far_11 = textureLod(coc_point_tx2D, tex_coord_11, 0).y;

    // top left
    float weight_01 = 1000.0;
    vec4 color_mul_coc_far = weight_01 * textureLod(color_point_tx2D, tex_coord_01, 0);
    float weight_sum = weight_01;

    // bottom left
    float weight_00 = 1.0 / (abs(coc_far_01 - coc_far_00) + 0.001);
    color_mul_coc_far += weight_00 * textureLod(color_point_tx2D, tex_coord_00, 0);
    weight_sum += weight_00;

    // bottom right
    float weight_10 = 1.0 / (abs(coc_far_01 - coc_far_10) + 0.001);
    color_mul_coc_far += weight_10 * textureLod(color_point_tx2D, tex_coord_10, 0);
    weight_sum += weight_10;

    // top right
    float weight_11 = 1.0 / (abs(coc_far_01 - coc_far_11) + 0.001);
    color_mul_coc_far += weight_11 * textureLod(color_point_tx2D, tex_coord_11, 0);
    weight_sum += weight_11;

    color_mul_coc_far /= weight_sum;
    color_mul_coc_far *= coc.y;


    imageStore(color_4_tx2D, pixel_coords, color);
    imageStore(coc_4_tx2D, pixel_coords, coc);
    imageStore(color_mul_coc_far_4_tx2D, pixel_coords, color_mul_coc_far);
}
