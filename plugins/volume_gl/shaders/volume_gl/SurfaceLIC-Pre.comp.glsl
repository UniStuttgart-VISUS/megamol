#version 430

#include "common/SurfaceLIC-Functions.inc.glsl"

#extension GL_ARB_compute_shader : enable

/* render target resolution */
uniform vec2 rt_resolution;

/* world space extents */
uniform vec3 origin;
uniform vec3 resolution;

/* input textures */
uniform highp sampler2D depth_tx2D;
uniform highp sampler2D normal_tx2D;
uniform highp sampler3D velocity_tx3D;

/* output image */
layout(rgba32f, binding = 0) writeonly uniform highp image2D velocity_target;

/* blocks for computation */
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    // Get pixel coordinates
    vec3 gID = gl_GlobalInvocationID.xyz;

    if (gID.x >= rt_resolution.x || gID.y >= rt_resolution.y) return;

    const ivec2 pixel_coords = ivec2(gID.xy);
    vec2 pixel_tex_coords = pixel_coords / rt_resolution;

    // Check for surface at this pixel
    vec3 velocity = vec3(0.0f);
    float magnitude = 0.0f;

    const float depth = texture(depth_tx2D, pixel_tex_coords).x;
    const vec3 normal = texture(normal_tx2D, pixel_tex_coords).xyz;

    if (depth >= 0.0f && depth < 1.0f) {
        // Get position in world space
        const vec4 world_pos = screen_to_world_space(pixel_tex_coords, depth);

        // Get velocity
        velocity = texture(velocity_tx3D, (world_pos.xyz - origin) / resolution).xyz;

        magnitude = length(velocity);

        // Project velocity onto the surface
        velocity -= dot(velocity, normal) * normal;
        velocity = ((proj_mx * view_mx) * vec4(velocity, 0.0f)).xyz;
    }

    imageStore(velocity_target, pixel_coords, vec4(velocity.xy, magnitude, 1.0f));
}
