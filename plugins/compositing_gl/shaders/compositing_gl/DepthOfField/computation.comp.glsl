/**
 * MegaMol
 * Copyright (c) 2024, MegaMol Dev Team
 * All rights reserved.
 */

#version 450

uniform sampler2D color_4_tx2D;
uniform sampler2D color_mul_coc_far_4_tx2D;
uniform sampler2D coc_4_tx2D;
uniform sampler2D coc_near_blurred_4_tx2D;

layout(binding = 0, r11f_g11f_b10f) writeonly uniform image2D near_4_tx2D;
layout(binding = 1, r11f_g11f_b10f) writeonly uniform image2D far_4_tx2D;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// https://github.com/maxest/MaxestFramework/blob/master/samples/dof/data/dof_ps.hlsl
static const vec2 offsets[] = {
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

vec3 Near(ivec2 coords) {
    vec3 result = texelFetch(color_4_tx2D, coords, 0); // TODO: pointSampler

    for(int i = 0; i < 48; ++i) {
        vec2 offset = kernel_scale * offsets[i];
        result += texelFetch(color_4_tx2D, coords + offset, 0); // TODO: linearSampler
    }

    return result / 49.0;
}

vec3 Far(ivec2 coords) {
    vec3 result = texelFetch(color_mul_coc_far_4_tx2D, coords, 0); // TODO: pointSampler
    float weight_sum = texelFetch(coc_4_tx2D, coords, 0); // TODO: pointSampler

    for(int i = 0; i < 48; ++i) {
        vec2 offset = kernel_scale * offsets[i];

        float coc_sample = texelFetch(coc_4_tx2D, coords + offset, 0).y; // far value, TODO: pointSampler
        vec3 sample = texelFetch(color_mul_coc_far_4_tx2D, coords + offset, 0); // TODO: linearSampler

        result += sample;
        weight_sum += coc_sample;
    }

    return result / weight_sum;
}

void main() {
    uvec3 gID = gl_GlobalInvocationID.xyz;
    ivec2 pixel_coords = ivec2(gID.xy);
    ivec2 tgt_resolution = imageSize(tgt_tx2D);

    if (pixel_coords.x >= tgt_resolution.x || pixel_coords.y >= tgt_resolution.y) {
        return;
    }

    float coc_near_blurred = texelFetch(coc_near_blurred_4_tx2D, pixel_coords, 0).x; // near value, TODO: pointSampler
    float coc_far = texelFetch(coc_4_tx2D, pixel_coords, 0).y; // far value, TODO: pointSampler

    vec3 near = vec3(0.0);
    if(coc_near_blurred > 0.0) {
        near = Near(pixel_coords);
    } else {
        near = texelFetch(color_4_tx2D, pixel_coords, 0);
    }

    vec3 far = vec3(0.0);
    if(coc_far > 0.0) {
        far = Far(pixel_coords);
    }

    imageStore(near_4_tx2D, pixel_coords, near);
    imageStore(far_4_tx2D, pixel_coords, far);
}
