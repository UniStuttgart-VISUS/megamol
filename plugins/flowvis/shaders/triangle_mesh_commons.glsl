/**
 * Calculate the normal
 */
vec3 calculate_normal(mat4 view_mx, vec4 vertex_1, vec4 vertex_2, vec4 vertex_3) {
    vec3 normal = normalize(cross(vertex_1.xyz - vertex_2.xyz, vertex_1.xyz - vertex_3.xyz));

    const vec3 view_dir = normalize((inverse(view_mx) * vec4(0.0f, 0.0f, -1.0f, 0.0f)).xyz);

    if (dot(view_dir, normal) > 0.0f) {
        normal *= -1.0f;
    }

    return normal;
}

/**
 * Perform halfspace test
 *
 * Return true if vertex is clipped; false otherwise
 */
bool clip_halfspace(vec3 vertex, vec4 plane) {
    return dot(plane, vec4(vertex, 1.0f)) > 0.0f;
}
