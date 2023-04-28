vec3 depthToWorldPos(float depth, vec2 uv, mat4 inverse_view_matrix, mat4 inverse_projection_matrix) {
    float z = depth * 2.0 - 1.0;

    vec4 cs_pos = vec4(uv * 2.0 - 1.0, z, 1.0);
    vec4 vs_pos = inverse_projection_matrix * cs_pos;

    // Perspective division
    vs_pos /= vs_pos.w;
    
    vec4 ws_pos = inverse_view_matrix * vs_pos;

    return ws_pos.xyz;
}
