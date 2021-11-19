#version 450

#extension GL_NV_mesh_shader : enable

#define ROOT_2 1.414213562f
#define WARP 16
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

out Point {
    flat vec4 pointColor;
    flat vec3 objPos;
    vec3 ray;
    vec3 tt;
    flat float rad;
    flat float sqrRad;
    float tf;
}
pp[];

void main() {
    uint g_idx = gl_GlobalInvocationID.x; // TODO Check size
    if (g_idx < num_points) {
        uint l_idx = gl_LocalInvocationID.x;

        vec3 objPos;
        float rad;
        vec4 pointColor;
        access_data(g_idx, objPos, pointColor, rad);

        vec3 oc_pos = objPos - camPos;
        float sqrRad = rad * rad;

        float dd = dot(oc_pos, oc_pos);

        float s = (sqrRad) / (dd);

        float vi = rad / sqrt(1.0f - s);

        vec3 vr = normalize(cross(oc_pos, camUp)) * vi;
        vec3 vu = normalize(cross(oc_pos, vr)) * vi;

        vec3 vd[NUM_V];
        vec4 v[NUM_V];
        v[0] = vec4(objPos - vr - vu, 1.0f);
        v[1] = vec4(objPos + vr - vu, 1.0f);
        v[2] = vec4(objPos + vr + vu, 1.0f);
        v[3] = vec4(objPos - vr + vu, 1.0f);

        for (int i = 0; i < NUM_V; ++i) {
            vd[i] = normalize(v[i].xyz - camPos);
            v[i] = MVP * v[i];
            v[i] /= v[i].w;
        }

        for (int i = 0; i < NUM_V; ++i) {
            pp[l_idx * NUM_V + i].sqrRad = sqrRad;
            pp[l_idx * NUM_V + i].ray = vd[i];
            pp[l_idx * NUM_V + i].tf = dot(oc_pos, vd[i]);
            pp[l_idx * NUM_V + i].tt = -oc_pos + pp[l_idx * NUM_V + i].tf * vd[i];

            pp[l_idx * NUM_V + i].objPos = objPos;
            pp[l_idx * NUM_V + i].rad = rad;
            pp[l_idx * NUM_V + i].pointColor = pointColor;

            gl_MeshVerticesNV[l_idx * NUM_V + i].gl_Position = v[i];
        }

        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 0] = l_idx * NUM_V + 0;
        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 1] = l_idx * NUM_V + 1;
        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 2] = l_idx * NUM_V + 2;

        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 3] = l_idx * NUM_V + 2;
        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 4] = l_idx * NUM_V + 3;
        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 5] = l_idx * NUM_V + 0;
    }
    gl_PrimitiveCountNV = min(num_points - gl_WorkGroupID.x * gl_WorkGroupSize.x, gl_WorkGroupSize.x) * NUM_P;
}


//#version 450
//
//#extension GL_NV_mesh_shader : enable
//
//#define ROOT_2 1.414213562f
//#define WARP 16
//#define NUM_V 9
//#define NUM_P 8
//
//layout(local_size_x = WARP) in;
//layout(max_vertices = WARP * NUM_V, max_primitives = WARP * NUM_P, triangles) out;
//
//#include "srtest_ubo.glsl"
//
//uniform uint num_points;
//
//uniform vec4 globalCol;
//uniform float globalRad;
//
//uniform bool useGlobalCol;
//uniform bool useGlobalRad;
//
//#include "srtest_ssbo.glsl"
//
//#include "srtest_touchplane.glsl"
//
//out Point {
//    flat vec4 pointColor;
//    flat vec3 objPos;
//    vec3 ray;
//    vec3 tt;
//    flat float rad;
//    flat float sqrRad;
//    float tf;
//}
//pp[];
//
//void main() {
//    uint g_idx = gl_GlobalInvocationID.x; // TODO Check size
//    if (g_idx < num_points) {
//        uint l_idx = gl_LocalInvocationID.x;
//
//        vec3 objPos;
//        float rad;
//        vec4 pointColor;
//        access_data(g_idx, objPos, pointColor, rad);
//
//        vec3 oc_pos = objPos - camPos;
//        float sqrRad = rad * rad;
//
//        float dd = dot(oc_pos, oc_pos);
//
//        float s = (sqrRad) / (dd);
//
//        float vi = rad / sqrt(1.0f - s);
//
//        vec3 vr = normalize(cross(oc_pos, camUp)) * vi;
//        vec3 vu = normalize(cross(oc_pos, vr)) * vi;
//
//        vec3 vd[NUM_V];
//        vec4 v[NUM_V];
//        v[0] = vec4(objPos - vr - vu, 1.0f);
//        v[1] = vec4(objPos - vu, 1.0f);
//        v[2] = vec4(objPos + vr - vu, 1.0f);
//        v[3] = vec4(objPos + vr, 1.0f);
//        v[4] = vec4(objPos + vr + vu, 1.0f);
//        v[5] = vec4(objPos + vu, 1.0f);
//        v[6] = vec4(objPos - vr + vu, 1.0f);
//        v[7] = vec4(objPos - vr, 1.0f);
//        v[8] = vec4(objPos, 1.0f);
//
//        for (int i = 0; i < NUM_V; ++i) {
//            vd[i] = normalize(v[i].xyz - camPos);
//            v[i] = MVP * v[i];
//            v[i] /= v[i].w;
//        }
//
//        for (int i = 0; i < NUM_V; ++i) {
//            pp[l_idx * NUM_V + i].sqrRad = sqrRad;
//            pp[l_idx * NUM_V + i].ray = vd[i];
//            pp[l_idx * NUM_V + i].tf = dot(oc_pos, vd[i]);
//            pp[l_idx * NUM_V + i].tt = -oc_pos + pp[l_idx * NUM_V + i].tf * vd[i];
//
//            pp[l_idx * NUM_V + i].objPos = objPos;
//            pp[l_idx * NUM_V + i].rad = rad;
//            pp[l_idx * NUM_V + i].pointColor = pointColor;
//
//            gl_MeshVerticesNV[l_idx * NUM_V + i].gl_Position = v[i];
//        }
//
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 0] = l_idx * NUM_V + 0;
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 1] = l_idx * NUM_V + 1;
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 2] = l_idx * NUM_V + 8;
//
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 3] = l_idx * NUM_V + 1;
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 4] = l_idx * NUM_V + 2;
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 5] = l_idx * NUM_V + 8;
//
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 6] = l_idx * NUM_V + 2;
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 7] = l_idx * NUM_V + 3;
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 8] = l_idx * NUM_V + 8;
//
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 9] = l_idx * NUM_V + 3;
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 10] = l_idx * NUM_V + 4;
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 11] = l_idx * NUM_V + 8;
//
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 12] = l_idx * NUM_V + 4;
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 13] = l_idx * NUM_V + 5;
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 14] = l_idx * NUM_V + 8;
//
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 15] = l_idx * NUM_V + 5;
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 16] = l_idx * NUM_V + 6;
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 17] = l_idx * NUM_V + 8;
//
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 18] = l_idx * NUM_V + 6;
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 19] = l_idx * NUM_V + 7;
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 20] = l_idx * NUM_V + 8;
//
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 21] = l_idx * NUM_V + 7;
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 22] = l_idx * NUM_V + 0;
//        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 23] = l_idx * NUM_V + 8;
//    }
//    gl_PrimitiveCountNV = min(num_points - gl_WorkGroupID.x * gl_WorkGroupSize.x, gl_WorkGroupSize.x) * NUM_P;
//}
