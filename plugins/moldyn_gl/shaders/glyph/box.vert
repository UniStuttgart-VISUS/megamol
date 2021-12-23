out vec4 ws_pos;
out vec4 vert_color;

out vec3 inv_rad;

out flat vec3 dir_color;
out flat vec3 normal;
out flat vec3 transformed_normal;
out vec3 view_ray;

// this re-activates the old code that draws the whole box and costs ~50% performance for box glyphs
// note that you need the single-strip-per-box instancing call in the Renderer for this (14 * #glyphs)
// the new code uses 4 * (3 * #glyphs)
//#define UNOPTIMIZED

void main() {

#ifdef UNOPTIMIZED
    uint inst = gl_InstanceID;
    uint corner = gl_VertexID;
    normal = cube_normals[corner];
    vec4 corner_pos = vec4(cube_strip[corner], 0.0);
#else
    uint inst = gl_InstanceID / 3;
    uint face = gl_InstanceID % 3;
    uint corner = gl_VertexID;
    normal = cube_face_normals[face];
    vec4 corner_pos = vec4(cube_faces[face][corner], 0.0);
#endif

    bool ignore_radii = (options & OPTIONS_IGNORE_RADII) > 0;
    vec4 in_pos = vec4(pos_array[inst].x, pos_array[inst].y, pos_array[inst].z, 1.0);
    vec3 radii = vec3(rad_array[inst].x, rad_array[inst].y, rad_array[inst].z); //rad[inst];

    if (ignore_radii) {
        radii = vec3(1.0);
    } else {
        // check radii if they are below the minimum
        radii.x = radii.x < min_radius ? min_radius : radii.x;
        radii.y = radii.y < min_radius ? min_radius : radii.y;
        radii.z = radii.z < min_radius ? min_radius : radii.z;
    }

    vec3 absradii = abs(radii) * scaling;
    vec4 quat_c = vec4(quat_array[inst].x, quat_array[inst].y, quat_array[inst].z, quat_array[inst].w); //quat[inst];
    inv_rad = 1.0 / absradii;

    mat3 rotate_world_into_tensor = quaternion_to_matrix(quat_c);
    mat3 rotate_points = transpose(rotate_world_into_tensor);
    mat3 rotate_vectors = rotate_points; //transpose(inverse(rotate_points)); // makes no difference

    transformed_normal = (rotate_vectors * normal).xyz;

#ifdef UNOPTIMIZED
#else
    // if our cube face looks away from the camera, we need to replace it with the opposing one
    vec3 view_vec = cam.xyz - in_pos.xyz;
    if (dot(view_vec, transformed_normal) < 0) {
        normal = -normal;
        transformed_normal = (rotate_vectors * normal).xyz;
        // the winding changes like this (I think), but we do not CULL_FACE anyway
        corner_pos = -corner_pos;
    }
#endif

    vec3 dir_color1 = max(vec3(0), normal * sign(radii));
    vec3 dir_color2 = vec3(1) + normal * sign(radii);
    dir_color = any(lessThan(dir_color2, vec3(0.5)))? dir_color2 * vec3(0.5) : dir_color1;

    uint flag = FLAG_ENABLED;
    bool flags_available = (options & OPTIONS_USE_FLAGS) > 0;
    if (flags_available) {
        flag = flags_array[(flag_offset + inst)];
    }

    vec4 col = vec4(col_array[inst].x, col_array[inst].y, col_array[inst].z, col_array[inst].w);

    bool use_per_axis_color = (options & OPTIONS_USE_PER_AXIS_COLOR) > 0;
    if(use_per_axis_color){
        col.r = face == 0 ? radii.x : face == 1 ? radii.y : radii.z;
    }

    vert_color = compute_color(col, flag, tf_texture, tf_range, global_color, flag_selected_col, flag_softselected_col, options);

    corner_pos.xyz *= absradii; // scale
    corner_pos.xyz = rotate_points * corner_pos.xyz;
    ws_pos = in_pos + corner_pos; // corners relative to world space glyph positions

    view_ray = cam.xyz - ws_pos.xyz;
    gl_Position =  mvp * ws_pos;

    bool clip_available = (options & OPTIONS_USE_CLIP) > 0;
    if (clip_available) {
        vec3 plane_normal = clip_data.xyz;
        float dist = dot(plane_normal, ws_pos.xyz) + clip_data.w;
        gl_ClipDistance[0] = dist;
    }

    if (flags_available) {
        if (!bitflag_isVisible(flag)) {
            gl_ClipDistance[0] = -1.0;
        } else {
            gl_ClipDistance[0] = 1.0;
        }
    }
}
