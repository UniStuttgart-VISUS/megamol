/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

out vec4 obj_pos;
out vec3 cam_pos;
out vec3 radii;
out vec3 absradii;
out mat3 rotate_tensor_into_world;

out vec4 vert_color;


// this re-activates the old code that draws the whole box and costs ~50% performance for box glyphs
// note that you need the single-strip-per-box instancing call in the Renderer for this (14 * #glyphs)
// the new code uses 4 * (3 * #glyphs)


void main() {
    // initial stuff
    uint inst = gl_InstanceID / 3;
    uint face = gl_InstanceID % 3;
    uint corner = gl_VertexID;
    vec4 corner_pos = vec4(cube_faces[face][corner], 0.0);


    // quaternion stuff
    vec4 quat_c = vec4(quat_array[inst].x, quat_array[inst].y, quat_array[inst].z, quat_array[inst].w); //quat[inst];
    // quaternion_to_matrix returns the transposed world_into_tensor rotation matrix
    rotate_tensor_into_world = quaternion_to_matrix(quat_c);
    mat3 rotate_world_into_tensor = transpose(rotate_tensor_into_world);


    // cube orientation
    obj_pos = vec4(pos_array[inst].x, pos_array[inst].y, pos_array[inst].z, 1.0);
    // shift, rotate, and scale cam
    vec3 view_vec = cam.xyz - obj_pos.xyz;
    vec3 normal = cube_face_normals[face];
    vec3 tensor_space_normal = (rotate_world_into_tensor * normal).xyz;

    // if our cube face looks away from the camera, we need to replace it with the opposing one
    if ( dot( view_vec, tensor_space_normal ) < 0 ) {
        // the winding changes like this (I think), but we do not CULL_FACE anyway
        corner_pos = -corner_pos;
    }

    // need to rotate by inverse tensor rotation in order to preserve relative
    // position between camera and transformed coordinate in fragment shader
    cam_pos = rotate_tensor_into_world * view_vec;


    // flags
    uint flag = FLAG_ENABLED;
    bool flags_available = (options & OPTIONS_USE_FLAGS) > 0;
    if (flags_available) {
        flag = flags_array[(flag_offset + inst)];
    }


    // radii stuff
    radii = vec3(rad_array[inst].x, rad_array[inst].y, rad_array[inst].z); //rad[inst];

    if( ( options & OPTIONS_IGNORE_RADII ) > 0 ) {
        radii = vec3(1.0);
    } else {
        // check radii if they are below the minimum
        // radii.x = radii.x < min_radius ? min_radius : radii.x;
        // radii.y = radii.y < min_radius ? min_radius : radii.y;
        // radii.z = radii.z < min_radius ? min_radius : radii.z;

        radii = max(radii, vec3(min_radius));
    }

    absradii = abs(radii) * scaling;


    // color stuff
    vec4 col = vec4(col_array[inst].x, col_array[inst].y, col_array[inst].z, col_array[inst].w);

    // TODO: what effect does this have to have for arrows?
    if( ( options & OPTIONS_USE_PER_AXIS_COLOR ) > 0 ) {
        col.r = face == 0 ? radii.x : face == 1 ? radii.y : radii.z;
    }

    vert_color = compute_color(col, flag, tf_texture, tf_range, global_color, flag_selected_col, flag_softselected_col, options);


    // vertex stuff
    corner_pos.xyz *= absradii; // scale
    corner_pos.xyz = rotate_world_into_tensor * corner_pos.xyz;

    // corners relative to tensor space glyph positions
    vec4 ts_pos = obj_pos + corner_pos;

    gl_Position =  mvp * ( ts_pos );


    // clipping
    if ( ( options & OPTIONS_USE_CLIP ) > 0 ) {
        vec3 plane_normal = clip_data.xyz;
        float dist = dot(plane_normal, ts_pos.xyz) + clip_data.w;
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
