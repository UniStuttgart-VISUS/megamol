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

// TODO: does this work as expected?
float DepthNDCToView(float depth_ndc) {
    return -proj_params.y / (depth_ndc + proj_params.x);
}

void main() {
    uvec3 gID = gl_GlobalInvocationID.xyz;
    ivec2 pixel_coords = ivec2(gID.xy);
    ivec2 tgt_res = imageSize(coc_tx2D);
    vec2 tex_coords = (vec2(pixel_coords) + vec2(0.5)) / vec2(tgt_res);

    float ndc_depth = texture(depth_point_tx2D, tex_coords).x;
    float view_depth = -DepthNDCToView(ndc_depth);

    float near_coc = 0.0;
    if(view_depth < fields[0]) {
        near_coc = 1.0;
    } else if( view_depth < fields[1]) {
        near_coc = (view_depth - fields[0]) / (fields[1] - fields[0]);
        near_coc = clamp(near_coc, 0.0, 1.0);
    }

    float far_coc = 0.0;
    if(far_coc < fields[2]) {
        far_coc = 1.0;
    } else if(far_coc < fields[3]) {
        far_coc = (view_depth - fields[2]) / (fields[3] - fields[2]);
        far_coc = clamp(far_coc, 0.0, 1.0);
    }


    imageStore(coc_tx2D, pixel_coords, vec4(near_coc, far_coc, 0.0, 0.0));
}
