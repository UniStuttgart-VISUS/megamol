out vec4 objPos;
out vec3 camPos;
out vec4 lightPos;
out vec4 vertColor;

out vec3 invRad;

out flat vec3 dirColor;
out flat vec3 transformedNormal;
out vec3 viewRay;

out mat3 rotate_world_into_tensor;
out mat3 rotate_points;

void main() {
    vec4 tmp, tmp1;
    vec3 tmp2;
    
    uint inst = gl_InstanceID / 3;
    uint face = gl_InstanceID % 3;
    uint corner = gl_VertexID;
    vec3 normal = cube_face_normals[face];
    vec4 cornerPos = vec4(cube_faces[face][corner], 0.0);

    vec4 inPos = vec4(posArray[inst].x, posArray[inst].y, posArray[inst].z, 1.0);
    vec3 radii = vec3(radArray[inst].x, radArray[inst].y, radArray[inst].z); //rad[inst];
    vec3 absradii = abs(radii) * scaling;
    vec4 quatC = vec4(quatArray[inst].x, quatArray[inst].y, quatArray[inst].z, quatArray[inst].w); //quat[inst];
    invRad = 1.0 / absradii;

    rotate_world_into_tensor = quaternion_to_matrix(quatC);
    rotate_points = transpose(rotate_world_into_tensor);
    //mat3 rotate_vectors = rotate_points; //transpose(inverse(rotate_points)); // makes no difference
    
    transformedNormal = (rotate_points * normal).xyz;

    // if our cube face looks away from the camera, we need to replace it with the opposing one
    vec3 view_vec = cam.xyz - inPos.xyz;
    if (dot(view_vec, transformedNormal) < 0) {
        normal = -normal;
        transformedNormal = (rotate_points * normal).xyz;
        // the winding changes like this (I think), but we do not CULL_FACE anyway
        cornerPos = -cornerPos;
    }

    objPos = inPos;
  
    tmp.xyz = cam.xyz - objPos.xyz;
    camPos.xyz = rotate_world_into_tensor * tmp.xyz;
    camPos.xyz *= invRad;

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

    cornerPos.xyz *= absradii; // scale
    cornerPos.xyz = rotate_points * cornerPos.xyz;
    vec4 wsPos = inPos + cornerPos; // corners relative to world space glyph positions

    viewRay = cam.xyz - wsPos.xyz;
    gl_Position =  MVP * wsPos;

    bool clip_available = (options & OPTIONS_USE_CLIP) > 0;
    if (clip_available) {
        vec3 planeNormal = clip_data.xyz;
        float dist = dot(planeNormal, wsPos.xyz) + clip_data.w;
        gl_ClipDistance[0] = dist;
    }

    if (flags_available && !bitflag_isVisible(flag)) {
        gl_ClipDistance[0] = -1.0;
    }

}