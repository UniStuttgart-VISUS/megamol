/**
 * MegaMol
 * Copyright (c) 2024, MegaMol Dev Team
 * All rights reserved.
 */

#version 450

uniform sampler2D coc_4_point_tx2D;
uniform sampler2D coc_near_blurred_4_point_tx2D;
uniform sampler2D near_field_4_point_tx2D;
uniform sampler2D far_field_4_point_tx2D;

layout(binding = 0, r11f_g11f_b10f) writeonly uniform image2D near_field_filled_4_tx2D;
layout(binding = 1, r11f_g11f_b10f) writeonly uniform image2D far_field_filled_4_tx2D;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    uvec3 gID = gl_GlobalInvocationID.xyz;
    ivec2 pixel_coords = ivec2(gID.xy);
    ivec2 tgt_res = imageSize(near_field_filled_4_tx2D);
    vec2 tex_coords = (vec2(pixel_coords) + vec2(0.5)) / vec2(tgt_res);
    vec2 pixel_size = vec2(1.0) / vec2(tgt_res);

    vec4 near_filled = textureLod(near_field_4_point_tx2D, tex_coords, 0);
    float coc_near_blurred = textureLod(coc_near_blurred_4_point_tx2D, tex_coords, 0).x;
    if(coc_near_blurred > 0.0)
    {
        for(int i = -1; i <= 1; ++i)
        {
            for(int j = -1; j <= 1; ++j)
            {
                vec2 offset_coords = tex_coords + vec2(i, j) * pixel_size;
                vec4 smpl = textureLod(near_field_4_point_tx2D, offset_coords, 0);
                near_filled = max(near_filled, smpl);
            }
        }
    }

    vec4 far_filled = textureLod(far_field_4_point_tx2D, tex_coords, 0);
    float coc_far = textureLod(coc_4_point_tx2D, tex_coords, 0).y;
    if(coc_far > 0.0)
    {
        for(int i = -1; i <= 1; ++i)
        {
            for(int j = -1; j <= 1; ++j)
            {
                vec2 offset_coords = tex_coords + vec2(i, j) * pixel_size;
                vec4 smpl = textureLod(far_field_4_point_tx2D, offset_coords, 0);
                far_filled = max(far_filled, smpl);
            }
        }
    }


    imageStore(near_field_filled_4_tx2D, pixel_coords, near_filled);
    imageStore(far_field_filled_4_tx2D, pixel_coords, far_filled);
}
