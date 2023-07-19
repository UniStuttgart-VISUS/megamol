#version 450

#extension GL_NV_mesh_shader : enable

#define ROOT_2 1.414213562f
//#define WARP 32
#define NUM_V 4
#define NUM_P 2

layout(local_size_x = WARP) in;
layout(max_vertices = WARP * NUM_V, max_primitives = WARP * NUM_P, triangles) out;

#include "srtest_ubo.glsl"

uniform uint num_points;

uniform vec4 globalCol;
uniform float globalRad;

uniform bool useGlobalCol;
uniform bool useGlobalRad;

#include "srtest_ssbo.glsl"

#include "srtest_touchplane.glsl"

#include "srtest_frustum.glsl"

out Point {
    flat vec4 pointColor;
    flat vec3 oc_pos;
    flat float rad;
    flat float sqrRad;
}
pp[];

void main() {
    uint g_idx = gl_GlobalInvocationID.x;
    if (g_idx < num_points) {
        uint l_idx = gl_LocalInvocationID.x;

        vec3 objPos;
        float rad;
        vec4 pointColor;
        access_data(g_idx, objPos, pointColor, rad);

        vec3 oc_pos = objPos - camPos;
        float sqrRad = rad * rad;

        mat4 v;
#ifdef __SRTEST_CAM_ALIGNED__
        touchplane(objPos, rad, oc_pos, v);
#else
        touchplane_old(objPos, rad, oc_pos, v);
#endif

        for (int i = 0; i < NUM_V; ++i) {
            pp[l_idx * NUM_V + i].pointColor = pointColor;

            pp[l_idx * NUM_V + i].oc_pos = oc_pos;

            pp[l_idx * NUM_V + i].rad = rad;
            pp[l_idx * NUM_V + i].sqrRad = sqrRad;

            gl_MeshVerticesNV[l_idx * NUM_V + i].gl_Position = v[i];
        }

        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 0] = l_idx * NUM_V + 1;
        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 1] = l_idx * NUM_V + 2;
        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 2] = l_idx * NUM_V + 0;

        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 3] = l_idx * NUM_V + 0;
        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 4] = l_idx * NUM_V + 2;
        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 5] = l_idx * NUM_V + 3;
    }
    gl_PrimitiveCountNV = min(num_points - gl_WorkGroupID.x * gl_WorkGroupSize.x, gl_WorkGroupSize.x) * NUM_P;
}
