/**
 * MegaMol
 * Copyright (c) 2024, MegaMol Dev Team
 * All rights reserved.
 */

#version 450

uniform float blend;

uniform sampler2D color_tx2D;
uniform sampler2D coc_tx2D;
uniform sampler2D coc_4_tx2D;
uniform sampler2D coc_near_blurred_4_tx2D;
uniform sampler2D near_fill_4_tx2D;
uniform sampler2D far_fill_4_tx2D;

layout(binding = 0, rgba32f) writeonly uniform image2D depth_of_field_tx2D; // TODO: layout qualifier correct?

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    uvec3 gID = gl_GlobalInvocationID.xyz;
    ivec2 pixel_coords = ivec2(gID.xy);
    ivec2 tgt_resolution = imageSize(tgt_tx2D);

    if (pixel_coords.x >= tgt_resolution.x || pixel_coords.y >= tgt_resolution.y) {
        return;
    }

    // far field
    vec3 result = texelFetch(color_tx2D, pixel_coords, 0);

    vec2 pixel_coord_00 = pixel_coords;
    vec2 pixel_coord_10 = pixel_coords + ivec2(1, 0); // TODO: correct?
    vec2 pixel_coord_01 = pixel_coords + ivec2(0, 1); // TODO: correct?
    vec2 pixel_coord_11 = pixel_coords + ivec2(1, 1); // TODO: correct?

    float coc_far = texelFetch(coc_tx2D, pixel_coords, 0).y; // TODO: pointSampler
    // vec4(top_left, top_right, bottom_left, bottom_right)
    vec4 coc_far_x4 = textureGather(coc_4_tx2D, pixel_coord_00, 1).xywz; // TODO: pointSampler, TODO: order of gather (.xywz) correct?
    vec4 coc_far_diffs = abs(vec4(coc_far) - coc_far_x4);

    // TODO: there might be a problem with sampling here
    // we use full resolution pixel_coords to sample a quarter resolution buffer
    vec3 dof_far_00 = texelFetch(far_fill_4_tx2D, pixel_coord_00, 0); // TODO: pointSampler
    vec3 dof_far_10 = texelFetch(far_fill_4_tx2D, pixel_coord_10, 0); // TODO: pointSampler
    vec3 dof_far_01 = texelFetch(far_fill_4_tx2D, pixel_coord_01, 0); // TODO: pointSampler
    vec3 dof_far_11 = texelFetch(far_fill_4_tx2D, pixel_coord_11, 0); // TODO: pointSampler

    // TODO: this breaks when only using pixel_coords, need to use uv texture coordinates
    // maybe there is still a solution only using pixel_corods
    vec2 image_coord = pixel_coords;
    vec2 fractional = frac(image_coord);
    float a = (1.0 - fractional.x) * (1.0 - fractional.y);
    float b = fractional.x * (1.0 - fractional.y);
    float c = (1.0 - fractional.x) * fractional.y;
    float d = fractional.x * fractional.y;

    vec3 dof_far = 0.0;
    float weight_sum = 0.0;

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

    result = lerp(result, dof_far, blend * coc_far);


    // near field
    float coc_near = texelFetch(coc_near_blurred_4_tx2D, pixel_coords, 0).x; // TODO: linearSampler
    vec3 dof_near = texelFetch(near_fill_4_tx2D, pixel_coords, 0); // TODO: linearSampler

    result = lerp(result, dof_near, blend * coc_near);


    imageStore(depth_of_field_tx2D, pixel_coords, vec4(result, 0.0));
}
