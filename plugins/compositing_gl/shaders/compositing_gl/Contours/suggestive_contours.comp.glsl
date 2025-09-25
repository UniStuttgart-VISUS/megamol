#version 460

uniform sampler2D color_tex_2D;
uniform sampler2D intensity_tex;

uniform int radius;
uniform float threshold;

layout(rgba16) writeonly uniform image2D target_tex;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;


bool isValley(ivec2 pixel_coords, ivec2 resolution) {

    float p_max = 0;
    float p_i = texelFetch(intensity_tex, pixel_coords, 0).x;
    float s = 1 - 1 / radius;

    int pixel_count = 0;
    int strictly_darker_count = 0;

    for(int i = pixel_coords.x - radius; i <= pixel_coords.x + radius; i++) {
        for(int j = pixel_coords.y - radius; j <= pixel_coords.y + radius; j++) {

            ivec2 current_pos = ivec2(i, j);
            float current_pos_intensity = texelFetch(intensity_tex, current_pos, 0).x;

            if(distance(pixel_coords, current_pos) < radius) {
                pixel_count++;

                if(p_max < current_pos_intensity) {
                    p_max = current_pos_intensity;
                }

                if(p_i < current_pos_intensity) {
                    strictly_darker_count++;
                }

            }
        } 
    } 

    if(strictly_darker_count / pixel_count < s && p_max - p_i > threshold) {
        return true;
    } else {
        return false;
    }
}

void main() {
    uvec3 gID = gl_GlobalInvocationID.xyz;
    ivec2 pixel_coords = ivec2(gID.xy);
    ivec2 target_res = imageSize(target_tex);

    if(pixel_coords.x > target_res.x || pixel_coords.y > target_res.y){
        return;
    }

    vec4 black = vec4(0, 0, 0, 1);
    vec4 color = texelFetch(color_tex_2D, pixel_coords, 0);

    // if(isValley(pixel_coords, target_res)) {
    //     imageStore(target_tex, pixel_coords, vec4(0, 0, 0, 1));
    // } else {
    //     imageStore(target_tex, pixel_coords, color);
    // }
    imageStore(target_tex, pixel_coords, vec4(texelFetch(intensity_tex, pixel_coords, 0).xyz, 1));
}

