/**
 * MegaMol
 * Copyright (c) 2024, MegaMol Dev Team
 * All rights reserved.
 */

#version 450

uniform sampler2D color_4_point_tx2D;
uniform sampler2D color_4_linear_tx2D;
uniform sampler2D color_mul_coc_far_4_point_tx2D;
uniform sampler2D color_mul_coc_far_4_linear_tx2D;
uniform sampler2D coc_4_point_tx2D;
uniform sampler2D coc_4_linear_tx2D;
uniform sampler2D coc_near_blurred_4_point_tx2D;

uniform float kernel_scale;

layout(binding = 0, r11f_g11f_b10f) writeonly uniform image2D near_field_4_tx2D;
layout(binding = 1, r11f_g11f_b10f) writeonly uniform image2D far_field_4_tx2D;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// https://github.com/maxest/MaxestFramework/blob/master/samples/dof/data/dof_ps.hlsl
const vec2 offsets[] = {
    // first ring
    2.0 * vec2(1.000000f, 0.000000f),
    2.0 * vec2(0.707107f, 0.707107f),
    2.0 * vec2(-0.000000f, 1.000000f),
    2.0 * vec2(-0.707107f, 0.707107f),
    2.0 * vec2(-1.000000f, -0.000000f),
    2.0 * vec2(-0.707106f, -0.707107f),
    2.0 * vec2(0.000000f, -1.000000f),
    2.0 * vec2(0.707107f, -0.707107f),

    // second ring
    4.0 * vec2(1.000000f, 0.000000f),
    4.0 * vec2(0.923880f, 0.382683f),
    4.0 * vec2(0.707107f, 0.707107f),
    4.0 * vec2(0.382683f, 0.923880f),
    4.0 * vec2(-0.000000f, 1.000000f),
    4.0 * vec2(-0.382684f, 0.923879f),
    4.0 * vec2(-0.707107f, 0.707107f),
    4.0 * vec2(-0.923880f, 0.382683f),
    4.0 * vec2(-1.000000f, -0.000000f),
    4.0 * vec2(-0.923879f, -0.382684f),
    4.0 * vec2(-0.707106f, -0.707107f),
    4.0 * vec2(-0.382683f, -0.923880f),
    4.0 * vec2(0.000000f, -1.000000f),
    4.0 * vec2(0.382684f, -0.923879f),
    4.0 * vec2(0.707107f, -0.707107f),
    4.0 * vec2(0.923880f, -0.382683f),

    // third ring
    6.0 * vec2(1.000000f, 0.000000f),
    6.0 * vec2(0.965926f, 0.258819f),
    6.0 * vec2(0.866025f, 0.500000f),
    6.0 * vec2(0.707107f, 0.707107f),
    6.0 * vec2(0.500000f, 0.866026f),
    6.0 * vec2(0.258819f, 0.965926f),
    6.0 * vec2(-0.000000f, 1.000000f),
    6.0 * vec2(-0.258819f, 0.965926f),
    6.0 * vec2(-0.500000f, 0.866025f),
    6.0 * vec2(-0.707107f, 0.707107f),
    6.0 * vec2(-0.866026f, 0.500000f),
    6.0 * vec2(-0.965926f, 0.258819f),
    6.0 * vec2(-1.000000f, -0.000000f),
    6.0 * vec2(-0.965926f, -0.258820f),
    6.0 * vec2(-0.866025f, -0.500000f),
    6.0 * vec2(-0.707106f, -0.707107f),
    6.0 * vec2(-0.499999f, -0.866026f),
    6.0 * vec2(-0.258819f, -0.965926f),
    6.0 * vec2(0.000000f, -1.000000f),
    6.0 * vec2(0.258819f, -0.965926f),
    6.0 * vec2(0.500000f, -0.866025f),
    6.0 * vec2(0.707107f, -0.707107f),
    6.0 * vec2(0.866026f, -0.499999f),
    6.0 * vec2(0.965926f, -0.258818f),
};

vec4 Near(vec2 coords, vec2 pixel_size) {
    vec4 result = textureLod(color_4_point_tx2D, coords, 0);

    for(int i = 0; i < 48; ++i) {
        vec2 offset = kernel_scale * offsets[i] * pixel_size;
        result += textureLod(color_4_linear_tx2D, coords + offset, 0);
    }

    return result / 49.0;
}

vec4 Far(vec2 coords, vec2 pixel_size) {
    vec4 result = textureLod(color_mul_coc_far_4_point_tx2D, coords, 0);
    float weight_sum = textureLod(coc_4_point_tx2D, coords, 0).y;

    for(int i = 0; i < 48; ++i) {
        vec2 offset = kernel_scale * offsets[i] * pixel_size;

        float coc_sample = textureLod(coc_4_linear_tx2D, coords + offset, 0).y; // far value
        vec4 smpl = textureLod(color_mul_coc_far_4_linear_tx2D, coords + offset, 0);

        result += smpl;
        weight_sum += coc_sample;
    }

    return result / weight_sum;
}

void main() {
    uvec3 gID = gl_GlobalInvocationID.xyz;
    ivec2 pixel_coords = ivec2(gID.xy);
    ivec2 tgt_res = imageSize(near_field_4_tx2D);
    vec2 tex_coords = (vec2(pixel_coords) + vec2(0.5)) / vec2(tgt_res);
    vec2 pixel_size = vec2(1.0) / vec2(tgt_res);

    float coc_near_blurred = textureLod(coc_near_blurred_4_point_tx2D, tex_coords, 0).x; // near value
    float coc_far = textureLod(coc_4_point_tx2D, tex_coords, 0).y; // far value

    vec4 near_field = vec4(0.0);
    if(coc_near_blurred > 0.0) {
        near_field = Near(tex_coords, pixel_size);
    } else {
        near_field = textureLod(color_4_point_tx2D, tex_coords, 0);
    }

    vec4 far_field = vec4(0.0);
    if(coc_far > 0.0) {
        far_field = Far(tex_coords, pixel_size);
    }

    imageStore(near_field_4_tx2D, pixel_coords, near_field);
    imageStore(far_field_4_tx2D, pixel_coords, far_field);
}
