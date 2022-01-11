void ellipsoid(in uint inst, in uint face, in uint corner, in vec4 objPos, in mat3 rotate_world_into_tensor,
    in mat3 rotate_points, in vec3 radii, out vec4 outPos, out vec4 outColor, out vec3 camPos,
    out vec3 dirColor, out vec3 invRad) {
    vec3 normal = cube_face_normals[face];
    vec4 cornerPos = vec4(cube_faces[face][corner], 0.0);

    vec3 absradii = abs(radii) * scaling;
    invRad = 1.0 / absradii;

    vec3 transformedNormal = (rotate_points * normal).xyz;

    // if our cube face looks away from the camera, we need to replace it with the opposing one
    vec3 view_vec = cam.xyz - objPos.xyz;
    if (dot(view_vec, transformedNormal) < 0) {
        normal = -normal;
        transformedNormal = (rotate_points * normal).xyz;
        // the winding changes like this (I think), but we do not CULL_FACE anyway
        cornerPos = -cornerPos;
    }

    camPos.xyz = rotate_world_into_tensor * view_vec.xyz;
    camPos.xyz *= invRad;

    vec3 dirColor1 = max(vec3(0), normal * sign(radii));
    vec3 dirColor2 = vec3(1) + normal * sign(radii);
    dirColor = any(lessThan(dirColor2, vec3(0.5))) ? dirColor2 * vec3(0.5) : dirColor1;

    uint flag = FLAG_ENABLED;
    bool flags_available = (options & OPTIONS_USE_FLAGS) > 0;
    if (flags_available) {
        flag = flagsArray[(flag_offset + inst)];
    }
    vec4 col = vec4(colArray[inst].x, colArray[inst].y, colArray[inst].z, colArray[inst].w);
    outColor =
        compute_color(col, flag, tf_texture, tf_range, global_color, flag_selected_col, flag_softselected_col, options);

    cornerPos.xyz *= absradii; // scale
    cornerPos.xyz = rotate_points * cornerPos.xyz;
    vec4 wsPos = objPos + cornerPos; // corners relative to world space glyph positions

    //outRay = cam.xyz - wsPos.xyz;
    outPos = MVP * wsPos;
}
