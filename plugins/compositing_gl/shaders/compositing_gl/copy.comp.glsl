/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#version 450

layout(local_size_x = 8, local_size_y = 8) in;

uniform sampler2D src_tx2D;
layout(rgba16f, binding = 0) writeonly uniform image2D tgt_tx2D;

void main() {
    ivec2 inPos = ivec2(gl_GlobalInvocationID.xy);

    vec4 value = texelFetch(src_tx2D, inPos, 0);

    imageStore(tgt_tx2D, inPos, value);
}
