/**
 * MegaMol
 * Copyright (c) 2024, MegaMol Dev Team
 * All rights reserved.
 */

#version 450

uniform sampler2D color_point_tx2D;
uniform sampler2D coc_point_tx2D;
uniform sampler2D coc_4_point_tx2D;
uniform sampler2D coc_near_blurred_4_linear_tx2D;
uniform sampler2D near_field_filled_4_linear_tx2D;
uniform sampler2D far_field_filled_4_point_tx2D;

uniform float blend;

layout(binding = 0, rgba32f) writeonly uniform image2D depth_of_field_tx2D; // TODO: layout qualifier correct? use outformat handler from host code

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    uvec3 gID = gl_GlobalInvocationID.xyz;
    ivec2 pixel_coords = ivec2(gID.xy);
    ivec2 tgt_res = imageSize(depth_of_field_tx2D);
    vec2 tex_coords = (vec2(pixel_coords) + vec2(0.5)) / vec2(tgt_res);
    vec2 pixel_size_4 = vec2(2.0) / vec2(tgt_res);

    // far field
    vec4 result = textureLod(color_point_tx2D, tex_coords, 0);

    vec2 tex_coord_00 = tex_coords;
    vec2 tex_coord_10 = tex_coords + vec2(pixel_size_4.x, 0);
    vec2 tex_coord_01 = tex_coords + vec2(0, pixel_size_4.y);
    vec2 tex_coord_11 = tex_coords + pixel_size_4;

    float coc_far = textureLod(coc_point_tx2D, tex_coords, 0).y;
    // vec4(top_left, top_right, bottom_left, bottom_right)
    // setting channel to one is equivalent to GatherGreen in hlsl
    vec4 coc_far_x4 = textureGather(coc_4_point_tx2D, tex_coord_00, 1).xywz; // TODO: is this correct? might need to offset tex_coord_00 with  + 0.5 * pixel_size_4 or smth like that
    vec4 coc_far_diffs = abs(vec4(coc_far) - coc_far_x4);

    vec4 dof_far_00 = textureLod(far_field_filled_4_point_tx2D, tex_coord_00, 0);
    vec4 dof_far_10 = textureLod(far_field_filled_4_point_tx2D, tex_coord_10, 0);
    vec4 dof_far_01 = textureLod(far_field_filled_4_point_tx2D, tex_coord_01, 0);
    vec4 dof_far_11 = textureLod(far_field_filled_4_point_tx2D, tex_coord_11, 0);

    vec2 image_coord = tex_coords / vec2(tgt_res);
    vec2 fractional = fract(image_coord);
    float a = (1.0 - fractional.x) * (1.0 - fractional.y);
    float b = fractional.x * (1.0 - fractional.y);
    float c = (1.0 - fractional.x) * fractional.y;
    float d = fractional.x * fractional.y;
    vec4 dof_far = vec4(0.0);
    float weight_sum = 0.0;

    // TODO: is assignment of a, b, c and d correct?
    float weight_01 = c / (coc_far_diffs.x + 0.001);
    dof_far += weight_01 * dof_far_01;
    weight_sum += weight_01;

    float weight_11 = d / (coc_far_diffs.y + 0.001);
    dof_far += weight_11 * dof_far_11;
    weight_sum += weight_11;

    float weight_00 = a / (coc_far_diffs.z + 0.001);
    dof_far += weight_00 * dof_far_00;
    weight_sum += weight_00;

    float weight_10 = b / (coc_far_diffs.w + 0.001);
    dof_far += weight_10 * dof_far_10;
    weight_sum += weight_10;

    dof_far /= weight_sum;

    result = mix(result, dof_far, blend * coc_far);


    // near field
    float coc_near = textureLod(coc_near_blurred_4_linear_tx2D, tex_coords, 0).x;
    vec4 dof_near = textureLod(near_field_filled_4_linear_tx2D, tex_coords, 0);

    result = mix(result, dof_near, blend * coc_near);


    imageStore(depth_of_field_tx2D, pixel_coords, vec4(result.xyz, 1.0));
}
