/* matrices */
uniform mat4 view_mx;
uniform mat4 proj_mx; 

/* transform screen to world space coordinates */
vec4 screen_to_world_space(vec2 screen_pos, float depth) {
    // Reconstruct clip space coordinates
    const vec3 clip_pos = vec3(screen_pos, depth) * 2.0f - vec3(1.0f);

    // Inverse transform to world space
    vec4 world_pos = inverse(proj_mx * view_mx) * vec4(clip_pos, 1.0f);
    world_pos /= world_pos.w;

    return world_pos;
}
