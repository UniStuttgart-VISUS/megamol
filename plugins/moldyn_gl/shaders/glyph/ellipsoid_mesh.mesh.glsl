#version 450

#extension GL_NV_mesh_shader : enable

#define WARP 16
//#define NUM_V 12
#define NUM_V 8
#define NUM_P 6

layout(local_size_x = WARP) in;
layout(max_vertices = WARP * NUM_V, max_primitives = WARP * NUM_P, triangles) out;

#include "glyph/uniforms.glsl"

#include "glyph/options.glsl"

#include "glyph/ssbo_data.glsl"

#include "core/tflookup.inc.glsl"

#include "glyph/flags.glsl"

#include "core/bitflags.inc.glsl"

#include "glyph/quaternion_to_matrix.glsl"

#include "glyph/cube_geometry.glsl"

#include "glyph/compute_color.glsl"

uniform uint num_points;

out Point {
    vec4 objPos;
    vec3 camPos;
    vec4 vertColor;

    vec3 invRad;

    flat vec3 dirColor;

    mat3 rotate_world_into_tensor;
    mat3 rotate_points;
}
pp[];


void ellipsoid(in uint inst, in uint face, in uint corner, in vec3 radii, in vec4 objPos,
    in mat3 rotate_world_into_tensor, in mat3 rotate_points, out vec3 dirColor, out vec3 invRad, out vec4 outPos,
    out vec3 camPos) {
    vec3 absradii = abs(radii) * scaling;
    invRad = 1.0 / absradii;

    //mat3 rotate_vectors = rotate_points; //transpose(inverse(rotate_points)); // makes no difference

    vec3 normal = cube_face_normals[face];
    vec3 transformedNormal = (rotate_points * normal).xyz;

    vec4 cornerPos = vec4(cube_faces[face][corner], 0.0);

    // if our cube face looks away from the camera, we need to replace it with the opposing one
    vec3 view_vec = cam.xyz - objPos.xyz;
    if (dot(view_vec, transformedNormal) < 0) {
        normal = -normal;
        transformedNormal = (rotate_points * normal).xyz;
        // the winding changes like this (I think), but we do not CULL_FACE anyway
        cornerPos = -cornerPos;
    }

    //tmp.xyz = cam.xyz - objPos.xyz;
    camPos.xyz = rotate_world_into_tensor * view_vec;
    camPos.xyz *= invRad;

    vec3 dirColor1 = max(vec3(0), normal * sign(radii));
    vec3 dirColor2 = vec3(1) + normal * sign(radii);
    dirColor = any(lessThan(dirColor2, vec3(0.5))) ? dirColor2 * vec3(0.5) : dirColor1;

    cornerPos.xyz *= absradii; // scale
    cornerPos.xyz = rotate_points * cornerPos.xyz;
    vec4 wsPos = objPos + cornerPos; // corners relative to world space glyph positions

    outPos = MVP * wsPos;
}


void main() {
    uint g_idx = gl_GlobalInvocationID.x;
    if (g_idx < num_points) {
        uint l_idx = gl_LocalInvocationID.x;

        uint inst = g_idx;

        vec4 inPos = vec4(posArray[inst].x, posArray[inst].y, posArray[inst].z, 1.0);
        vec4 quatC = vec4(quatArray[inst].x, quatArray[inst].y, quatArray[inst].z, quatArray[inst].w); //quat[inst];
        vec3 radii = vec3(radArray[inst].x, radArray[inst].y, radArray[inst].z);                       //rad[inst];
        vec4 col = vec4(colArray[inst].x, colArray[inst].y, colArray[inst].z, colArray[inst].w);

        mat3 rotate_world_into_tensor = quaternion_to_matrix(quatC);
        mat3 rotate_points = transpose(rotate_world_into_tensor);

        uint flag = FLAG_ENABLED;
        bool flags_available = (options & OPTIONS_USE_FLAGS) > 0;
        if (flags_available) {
            flag = flagsArray[(flag_offset + inst)];
        }

        vec4 vertColor = compute_color(
            col, flag, tf_texture, tf_range, global_color, flag_selected_col, flag_softselected_col, options);

        vec3 absradii = abs(radii) * scaling;
        vec3 invRad = 1.0 / absradii;

        vec3 view_vec = cam.xyz - inPos.xyz;
        vec3 camPos = rotate_world_into_tensor * view_vec;
        camPos *= invRad;

        for (uint vert = 0; vert < 8; ++vert) {
            uint v_idx = l_idx * NUM_V + vert;

            pp[v_idx].rotate_world_into_tensor = rotate_world_into_tensor;
            pp[v_idx].rotate_points = rotate_points;

            pp[v_idx].vertColor = vertColor;

            pp[v_idx].objPos = inPos;

            pp[v_idx].camPos = camPos;

            pp[v_idx].dirColor = vec3(1.0f);
            pp[v_idx].invRad = invRad;

            vec4 cornerPos = vec4(base_cube_verts[v_idx], 0.0);

            cornerPos.xyz *= absradii; // scale
            cornerPos.xyz = rotate_points * cornerPos.xyz;
            vec4 wsPos = inPos + cornerPos; // corners relative to world space glyph positions

            gl_MeshVerticesNV[v_idx].gl_Position = MVP * wsPos;
        }

        for (uint face = 0; face < 3; ++face) {
            uint base_v_idx = l_idx * NUM_V;
            uint base_p_idx = l_idx * 3 * NUM_P + face * 6;

            vec3 normal = cube_face_normals[face];
            vec3 transformedNormal = (rotate_points * normal).xyz;

            uvec3 f0, f1;
            if (dot(view_vec, transformedNormal) < 0) {
                f0 = face_ind_back[face][0];
                f1 = face_ind_back[face][1];
                // back
            } else {
                f0 = face_ind_front[face][0];
                f1 = face_ind_front[face][1];
                // front
            }

            gl_PrimitiveIndicesNV[base_p_idx + 0] = base_v_idx + f0.x;
            gl_PrimitiveIndicesNV[base_p_idx + 1] = base_v_idx + f0.y;
            gl_PrimitiveIndicesNV[base_p_idx + 2] = base_v_idx + f0.z;

            gl_PrimitiveIndicesNV[base_p_idx + 3] = base_v_idx + f1.x;
            gl_PrimitiveIndicesNV[base_p_idx + 4] = base_v_idx + f1.y;
            gl_PrimitiveIndicesNV[base_p_idx + 5] = base_v_idx + f1.z;
        }

#if 0
        for (uint face = 0; face < 3; ++face) {
            uint base_v_idx = l_idx * NUM_V + face * 4;
            for (uint corner = 0; corner < 4; ++corner) {
                uint v_idx = base_v_idx + corner;

                /*pp[v_idx].rotate_world_into_tensor = quaternion_to_matrix(quatC);
                pp[v_idx].rotate_points = transpose(pp[v_idx].rotate_world_into_tensor);*/
                pp[v_idx].rotate_world_into_tensor = rotate_world_into_tensor;
                pp[v_idx].rotate_points = rotate_points;
       
                /*pp[v_idx].vertColor = compute_color(
                    col, flag, tf_texture, tf_range, global_color, flag_selected_col, flag_softselected_col, options);*/
                pp[v_idx].vertColor = vertColor;

                pp[v_idx].objPos = inPos;

                vec4 outPos;

                ellipsoid(inst, face, corner, radii, pp[v_idx].objPos, pp[v_idx].rotate_world_into_tensor,
                    pp[v_idx].rotate_points, pp[v_idx].dirColor, pp[v_idx].invRad, outPos, pp[v_idx].camPos);

                pp[v_idx].camPos = camPos;
                pp[v_idx].invRad = invRad;

                gl_MeshVerticesNV[v_idx].gl_Position = outPos;


                /*bool clip_available = (options & OPTIONS_USE_CLIP) > 0;
                if (clip_available) {
                    vec3 planeNormal = clip_data.xyz;
                    float dist = dot(planeNormal, wsPos.xyz) + clip_data.w;
                    gl_ClipDistance[0] = dist;
                }

                if (flags_available && !bitflag_isVisible(flag)) {
                    gl_ClipDistance[0] = -1.0;
                }*/
            }
            /*uint base_p_idx = l_idx * 3 * NUM_P + face * 6;

            gl_PrimitiveIndicesNV[base_p_idx + 0] = base_v_idx + 0;
            gl_PrimitiveIndicesNV[base_p_idx + 1] = base_v_idx + 1;
            gl_PrimitiveIndicesNV[base_p_idx + 2] = base_v_idx + 2;

            gl_PrimitiveIndicesNV[base_p_idx + 3] = base_v_idx + 1;
            gl_PrimitiveIndicesNV[base_p_idx + 4] = base_v_idx + 3;
            gl_PrimitiveIndicesNV[base_p_idx + 5] = base_v_idx + 2;*/
        }

        /*for (uint face = 0; face < 3; ++face) {
            uint base_v_idx = l_idx * NUM_V + face * 4;
            uint base_p_idx = l_idx * 3 * NUM_P + face * 6;

            gl_PrimitiveIndicesNV[base_p_idx + 0] = base_v_idx + 0;
            gl_PrimitiveIndicesNV[base_p_idx + 1] = base_v_idx + 1;
            gl_PrimitiveIndicesNV[base_p_idx + 2] = base_v_idx + 2;

            gl_PrimitiveIndicesNV[base_p_idx + 3] = base_v_idx + 1;
            gl_PrimitiveIndicesNV[base_p_idx + 4] = base_v_idx + 3;
            gl_PrimitiveIndicesNV[base_p_idx + 5] = base_v_idx + 2;
        }*/

        uint base_v_idx = l_idx * NUM_V;
        uint base_p_idx = l_idx * 3 * NUM_P;
        gl_PrimitiveIndicesNV[base_p_idx + 0] = base_v_idx + 0;
        gl_PrimitiveIndicesNV[base_p_idx + 1] = base_v_idx + 1;
        gl_PrimitiveIndicesNV[base_p_idx + 2] = base_v_idx + 2;

        gl_PrimitiveIndicesNV[base_p_idx + 3] = base_v_idx + 1;
        gl_PrimitiveIndicesNV[base_p_idx + 4] = base_v_idx + 3;
        gl_PrimitiveIndicesNV[base_p_idx + 5] = base_v_idx + 2;

        gl_PrimitiveIndicesNV[base_p_idx + 6] = base_v_idx + 4;
        gl_PrimitiveIndicesNV[base_p_idx + 7] = base_v_idx + 5;
        gl_PrimitiveIndicesNV[base_p_idx + 8] = base_v_idx + 6;

        gl_PrimitiveIndicesNV[base_p_idx + 9] = base_v_idx + 5;
        gl_PrimitiveIndicesNV[base_p_idx + 10] = base_v_idx + 7;
        gl_PrimitiveIndicesNV[base_p_idx + 11] = base_v_idx + 6;

        gl_PrimitiveIndicesNV[base_p_idx + 12] = base_v_idx + 8;
        gl_PrimitiveIndicesNV[base_p_idx + 13] = base_v_idx + 9;
        gl_PrimitiveIndicesNV[base_p_idx + 14] = base_v_idx + 10;

        gl_PrimitiveIndicesNV[base_p_idx + 15] = base_v_idx + 9;
        gl_PrimitiveIndicesNV[base_p_idx + 16] = base_v_idx + 11;
        gl_PrimitiveIndicesNV[base_p_idx + 17] = base_v_idx + 10;
#endif
    }
    gl_PrimitiveCountNV = min(num_points - gl_WorkGroupID.x * gl_WorkGroupSize.x, gl_WorkGroupSize.x) * NUM_P;
}
