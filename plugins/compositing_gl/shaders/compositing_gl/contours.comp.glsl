#version 460

uniform sampler2D normal_tex_2D;
uniform sampler2D color_tex_2D;
uniform sampler2D depth_tex_2D;

uniform vec3 cam_pos;

uniform mat4 inv_view_mx;
uniform mat4 inv_proj_mx;

uniform float threshold = 1.0;

layout(rgba16) writeonly uniform image2D target_tex;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

mat3 sobel_kernel_y = mat3( 
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
);

mat3 sobel_kernel_x = mat3(
    -1, -2, -1,
    0,  0,  0,
    1,  2,  1
);

float applyFilter(mat3 kernel, ivec2 border, ivec2 pos) {
    float gradient = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {

            ivec2 neighbor_pos = pos + ivec2(i, j);
                vec4 neighbor_color = texelFetch(depth_tex_2D, neighbor_pos, 0); 
                gradient += neighbor_color.x * kernel[i+1][j+1];
        }
    }
    return gradient;
}


void main() {
    uvec3 gID = gl_GlobalInvocationID.xyz;
    ivec2 pixel_coords = ivec2(gID.xy);
    ivec2 target_res = imageSize(target_tex);

    if(pixel_coords.x > target_res.x || pixel_coords.y > target_res.y){
        return;
    }

    float gx = applyFilter(sobel_kernel_x, target_res, pixel_coords);
    float gy = applyFilter(sobel_kernel_y, target_res, pixel_coords);

    float gradient = sqrt(pow(gx, 2) + pow(gy, 2));
    vec4 color_texture = texelFetch(color_tex_2D, pixel_coords, 0);

    if(gradient > threshold) {
        imageStore(target_tex, pixel_coords, vec4(0));
    } else {
        imageStore(target_tex, pixel_coords, color_texture);
    }
}
