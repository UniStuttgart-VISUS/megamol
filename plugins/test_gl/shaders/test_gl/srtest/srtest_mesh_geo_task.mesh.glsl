#version 450

#extension GL_NV_mesh_shader : enable
#extension GL_NV_gpu_shader5 : enable

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

taskNV in Task {
    uint baseID;
    uint8_t subIDs[WARP];
}
IN;

out Point {
    flat vec4 pointColor;
    flat vec3 oc_pos;
    flat float rad;
    flat float sqrRad;
}
pp[];

void main() {
    uint g_idx = IN.baseID + IN.subIDs[gl_WorkGroupID.x];
    //uint g_idx = gl_GlobalInvocationID.x;
    //if (g_idx < num_points) {
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

        vec4 v[NUM_V];
        v[0] = vec4(objPos - vr - vu, 1.0f);
        v[1] = vec4(objPos + vr - vu, 1.0f);
        v[2] = vec4(objPos + vr + vu, 1.0f);
        v[3] = vec4(objPos - vr + vu, 1.0f);

        vec4 projPos = MVP * vec4(objPos + rad * (camDir), 1.0f);
        projPos = projPos / projPos.w;

        for (int i = 0; i < NUM_V; ++i) {
            v[i] = MVP * v[i];
            v[i] /= v[i].w;

            pp[l_idx * NUM_V + i].pointColor = pointColor;

            pp[l_idx * NUM_V + i].oc_pos = oc_pos;

            pp[l_idx * NUM_V + i].rad = rad;
            pp[l_idx * NUM_V + i].sqrRad = sqrRad;


            gl_MeshVerticesNV[l_idx * NUM_V + i].gl_Position = vec4(v[i].xy, projPos.z, 1.0f);
        }

        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 0] = l_idx * NUM_V + 1;
        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 1] = l_idx * NUM_V + 2;
        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 2] = l_idx * NUM_V + 0;

        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 3] = l_idx * NUM_V + 0;
        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 4] = l_idx * NUM_V + 2;
        gl_PrimitiveIndicesNV[l_idx * 3 * NUM_P + 5] = l_idx * NUM_V + 3;
    //}
        gl_PrimitiveCountNV = WARP * NUM_P;
}
