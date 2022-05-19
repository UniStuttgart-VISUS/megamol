#version 430

/**
 * Calculate the normal
 */
vec3 calculate_normal(vec4 vertex_1, vec4 vertex_2, vec4 vertex_3) {
    return normalize(cross(vertex_2.xyz - vertex_1.xyz, vertex_3.xyz - vertex_1.xyz));
}

/**
 * Is this triangle front facing
 *
 * Return true if front facing; false otherwise
 */
bool is_front_face(mat4 view_mx, vec3 normal) {
    const vec3 view_dir = normalize((inverse(view_mx) * vec4(0.0f, 0.0f, -1.0f, 0.0f)).xyz);

    return dot(view_dir, normal) < 0.0f;
}

/**
 * Perform halfspace test
 *
 * Return true if vertex is clipped; false otherwise
 */
bool clip_halfspace(vec3 vertex, vec4 plane) {
    return dot(plane, vec4(vertex, 1.0f)) > 0.0f;
}
