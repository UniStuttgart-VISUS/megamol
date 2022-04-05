/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */
#version 450

layout(local_size_x = 8, local_size_y = 8) in;

//-----------------------------------------------------------------------------
// UNIFORMS
uniform sampler2D g_input_tx2D;
layout(rgba8, binding = 0) uniform writeonly image2D g_output_tx2D;

void main() {
    ivec2 in_pos = ivec2(gl_GlobalInvocationID.xy);

    vec4 a = texelFetch(g_input_tx2D, 2 * in_pos + ivec2(0, 0), 0);
    vec4 b = texelFetch(g_input_tx2D, 2 * in_pos + ivec2(1, 0), 0);
    vec4 c = texelFetch(g_input_tx2D, 2 * in_pos + ivec2(0, 1), 0);
    vec4 d = texelFetch(g_input_tx2D, 2 * in_pos + ivec2(1, 1), 0);

    vec4 result = (a + b + c + d) / 4.f;

    imageStore(g_output_tx2D, in_pos, result);
}
