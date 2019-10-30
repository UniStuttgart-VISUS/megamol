/**
 * Calculate diffuse illumination coefficient for fixed light sources, adding an ambient term
 */
float calculate_illumination(mat4 view_mx, vec4 vertex_1, vec4 vertex_2, vec4 vertex_3) {
    vec3 normal = normalize(cross(vertex_1.xyz - vertex_2.xyz, vertex_1.xyz - vertex_3.xyz));

    const vec3 view_dir = normalize((inverse(view_mx) * vec4(0.0f, 0.0f, -1.0f, 0.0f)).xyz);

    if (dot(view_dir, normal) > 0.0f) {
        normal *= -1.0f;
    }

    const float ambient = 0.3f;
    const float diffuse = 0.7f;

    return min(1.0f, ambient + diffuse * dot(normal, -view_dir));
}
