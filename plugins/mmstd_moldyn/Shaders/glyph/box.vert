out vec4 wsPos;
out vec4 vertColor;

out vec3 invRad;

out flat vec3 dirColor;
out flat vec3 normal;
out flat vec3 transformedNormal;

void main() {
    uint inst = gl_InstanceID; //gl_VertexID / 14;
    uint corner = gl_VertexID;// % 14;

    vec4 inPos = vec4(posArray[inst].x, posArray[inst].y, posArray[inst].z, 1.0);
    vec3 radii = vec3(radArray[inst].x, radArray[inst].y, radArray[inst].z); //rad[inst];
    vec3 absradii = abs(radii);
    vec4 quatC = vec4(quatArray[inst].x, quatArray[inst].y, quatArray[inst].z, quatArray[inst].w); //quat[inst];
    invRad = 1.0 / absradii;
    
    mat3 rotate_world_into_tensor = quaternion_to_matrix(quatC);
    mat3 rotate_points = transpose(rotate_world_into_tensor);
    mat3 rotate_vectors = rotate_points; //transpose(inverse(rotate_points));

    normal = cube_normals[corner];
    transformedNormal = (rotate_vectors * normal).xyz;

    vec3 dirColor1 = max(vec3(0), normal * sign(radii));
    vec3 dirColor2 = vec3(1) + normal * sign(radii);
    dirColor = any(lessThan(dirColor2, vec3(0.5)))? dirColor2 * vec3(0.5) : dirColor1;

    uint flag = FLAG_ENABLED;
    bool flags_available = (options & OPTIONS_USE_FLAGS) > 0;
    if (flags_available) {
        flag = flagsArray[(flag_offset + inst)];
    }
    vec4 col = vec4(colArray[inst].x, colArray[inst].y, colArray[inst].z, colArray[inst].w);
    vertColor = compute_color(col, flag, tf_texture, tf_range, global_color, flag_selected_col, flag_softselected_col, options);
    
    vec4 cornerPos;
    cornerPos.xyz = cube_strip[corner];
    cornerPos.xyz *= absradii; // scale
    cornerPos.xyz = rotate_points * cornerPos.xyz;
    cornerPos.w = 0.0;
    wsPos = inPos + cornerPos; // corners relative to world space glyph positions

    gl_Position =  MVP * wsPos;

    bool clip_available = (options & OPTIONS_USE_CLIP) > 0;
    if (clip_available) {
        vec3 planeNormal = clip_data.xyz;
        float dist = dot(planeNormal, inPos.xyz) + clip_data.w;
        if (dist < 0.0) {
            gl_Position.w = 0.0;
        }
    }

    if (flags_available && !bitflag_isVisible(flag)) {
        gl_Position.w = 0.0;
    }
}