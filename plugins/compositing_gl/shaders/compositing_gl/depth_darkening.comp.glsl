#version 460

uniform sampler2D color_tex;
uniform sampler2D depth_tex;
uniform sampler2D blurred_depth_tex;

// only one of these 3 should be active
#if defined OUT32F
layout(rgba32f) writeonly uniform image2D target_tex;
#endif
#if defined OUT16HF
layout(rgba16f) writeonly uniform image2D target_tex;
#endif
#if defined OUT8NB
layout(rgba8_snorm) writeonly uniform image2D target_tex;
#endif

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

uniform float lambda = 1.0;

void main() {
    uvec3 gID = gl_GlobalInvocationID.xyz;
    ivec2 pixel_coords = ivec2(gID.xy);
    ivec2 target_res = imageSize(target_tex);

    if(pixel_coords.x > target_res.x || pixel_coords.y > target_res.y){
        return;
    }

    vec4 col = texelFetch(color_tex, pixel_coords, 0);
    vec4 dep = texelFetch(depth_tex, pixel_coords, 0);
    vec4 blr = texelFetch(blurred_depth_tex, pixel_coords, 0);

    vec4 delta = blr - dep; // delta D = G * D - D
    vec4 delta_minus = min(delta, vec4(0.0)); // negative part of delta D
    vec4 result = col + lambda * delta_minus; // I' = I + lambda * delta D-

    imageStore(target_tex, pixel_coords, vec4(result));
}
