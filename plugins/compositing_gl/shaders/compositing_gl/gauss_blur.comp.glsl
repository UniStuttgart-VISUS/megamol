#version 460

uniform sampler2D source_tex;

layout(rgba16) writeonly uniform image2D target_tex;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(std430, binding = 1) readonly buffer KernelBuffer { float kernel[]; };

uniform int kernel_radius;
uniform ivec2 kernel_direction;

void main() {
    uvec3 gID = gl_GlobalInvocationID.xyz;
    ivec2 pixel_coords = ivec2(gID.xy);
    ivec2 target_res = imageSize(target_tex);

    if(pixel_coords.x >= target_res.x || pixel_coords.y >= target_res.y || pixel_coords.x < 0 || pixel_coords.y < 0){
        return;
    }

    int s_val = kernel_radius - 1;
    vec4 col = texelFetch(source_tex, pixel_coords, 0);
    col = vec4(0.0);

    for(int i = -s_val; i <= s_val; ++i) {
        int k_pos = i + s_val;
        float k_val = kernel[k_pos];
        ivec2 coords = pixel_coords + i * kernel_direction;
        // texture wrapping does not work with texelFetch, so we have to clamp the coords here
        coords = clamp(coords, ivec2(0,0), ivec2(target_res.x - 1, target_res.y - 1));
        vec4 local_col = texelFetch(source_tex, coords, 0);
        col = col + local_col * k_val;
    }

    imageStore(target_tex, pixel_coords, vec4(col));
}
