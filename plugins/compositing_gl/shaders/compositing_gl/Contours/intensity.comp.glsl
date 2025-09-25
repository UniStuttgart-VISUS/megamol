#version 460

uniform sampler2D depth_tex_2D;
uniform sampler2D normal_tex_2D;

uniform vec3 cam_pos;

uniform mat4 inv_view_mx;
uniform mat4 inv_proj_mx;

layout(rgba16) writeonly uniform image2D target_tex;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

vec3 getViewVector(ivec2 pos, ivec2 resolution) {
    vec2 normal_pos = vec2(float(pos.x) / float(resolution.x), float(pos.y) / float(resolution.y));
    float depth = texelFetch(depth_tex_2D, pos, 0).x;
    float z = depth * 2.0 - 1.0;
    
    vec4 clipSpacePosition = vec4(normal_pos * 2.0 - 1.0, z, 1.0);
    vec4 viewSpacePosition = inv_proj_mx * clipSpacePosition;

    // Perspective division
    viewSpacePosition /= viewSpacePosition.w;

    vec4 worldSpacePosition = inv_view_mx * viewSpacePosition;

    return normalize(cam_pos - worldSpacePosition.xyz);
}


float getIntensity(ivec2 pos, ivec2 resolution) {
    vec3 view = getViewVector(pos, resolution);
    vec3 normal = texelFetch(normal_tex_2D, pos, 0).xyz;
    return dot(normalize(normal), view);
}

void main() {
    uvec3 gID = gl_GlobalInvocationID.xyz;
    ivec2 pixel_coords = ivec2(gID.xy);
    ivec2 target_res = imageSize(target_tex);

    if(pixel_coords.x > target_res.x || pixel_coords.y > target_res.y){
        return;
    }

    // imageStore(target_tex, pixel_coords, vec4(getIntensity(pixel_coords, target_res)));
    // imageStore(target_tex, pixel_coords, vec4(getViewVector(pixel_coords, target_res), 1));
    imageStore(target_tex, pixel_coords, vec4((texelFetch(normal_tex_2D, pixel_coords, 0).xyz), 1.0));
}