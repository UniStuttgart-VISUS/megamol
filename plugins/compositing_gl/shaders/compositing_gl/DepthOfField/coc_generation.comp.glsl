/**
 * MegaMol
 * Copyright (c) 2024, MegaMol Dev Team
 * All rights reserved.
 */

#version 450

uniform sampler2D depth_point_tx2D;
uniform vec2 proj_params;
uniform vec4 fields; // vec4(ne, nb, fb, fe)

layout(binding = 0, rg8) writeonly uniform image2D coc_tx2D;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;


float DepthNDCToView(float depth_ndc) {
    return -proj_params.y / (depth_ndc + proj_params.x);
}

void main() {
    uvec3 gID = gl_GlobalInvocationID.xyz;
    ivec2 pixel_coords = ivec2(gID.xy);
    ivec2 tgt_resolution = imageSize(tgt_tx2D);

    if (pixel_coords.x >= tgt_resolution.x || pixel_coords.y >= tgt_resolution.y) {
        return;
    }

    float ndc_depth = texelFetch(depth_point_tx2D, pixel_coords);
    float view_depth = DepthNDCToView(ndc_depth);
    float near = (fields[0] - view_depth) / (fields[0] - fields[1]);
    float far  = (view_depth - fields[2]) / (fields[3] - fields[2]);

    imageStore(coc_tx2D, pixel_coords, vec2(near, far));
}
