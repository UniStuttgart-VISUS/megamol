#version 450

#extension GL_NV_mesh_shader : enable

layout(local_size_x = WARP) in;
layout(max_vertices = WARP, max_primitives = WARP, points) out;

#include "srtest_ubo.glsl"

uniform uint num_points;

uniform vec4 globalCol;
uniform float globalRad;

uniform bool useGlobalCol;
uniform bool useGlobalRad;

#include "srtest_ssbo.glsl"

#include "srtest_touchplane.glsl"

out Point {
    flat vec3 objPos;
    flat float rad;
    flat float sqrRad;
    flat vec4 pointColor;
    flat vec3 oc_pos;
    flat float c;
    flat out vec4 ve0;
    flat out vec4 ve1;
    flat out vec4 ve2;
    flat out vec4 ve3;
    flat out vec2 vb;
    flat out float l;
}
pp[];

void main() {
    uint g_idx = gl_GlobalInvocationID.x;
    if (g_idx < num_points) {
        uint l_idx = gl_LocalInvocationID.x;

        access_data(g_idx, pp[l_idx].objPos, pp[l_idx].pointColor, pp[l_idx].rad);

        pp[l_idx].oc_pos = pp[l_idx].objPos - camPos;
        pp[l_idx].sqrRad = pp[l_idx].rad * pp[l_idx].rad;

        vec2 mins, maxs;

        float dd = dot(pp[l_idx].oc_pos, pp[l_idx].oc_pos);

        float s = (pp[l_idx].sqrRad) / (dd);

        float vi = pp[l_idx].rad / sqrt(1.0f - s);

        vec3 vr = normalize(cross(pp[l_idx].oc_pos, camUp)) * vi;
        vec3 vu = normalize(cross(pp[l_idx].oc_pos, vr)) * vi;

        vec4 v0 = vec4(pp[l_idx].objPos + vr, 1.0f);
        vec4 v1 = vec4(pp[l_idx].objPos - vr, 1.0f);
        vec4 v2 = vec4(pp[l_idx].objPos + vu, 1.0f);
        vec4 v3 = vec4(pp[l_idx].objPos - vu, 1.0f);

        pp[l_idx].ve0 = vec4(normalize((v1.xyz - vu) - camPos), 0.0);
        pp[l_idx].ve1 = vec4(normalize((v0.xyz - vu) - camPos), 0.0);
        pp[l_idx].ve2 = vec4(normalize((v0.xyz + vu) - camPos), 0.0);
        pp[l_idx].ve3 = vec4(normalize((v1.xyz + vu) - camPos), 0.0);

        pp[l_idx].ve0.w = dot(pp[l_idx].oc_pos, pp[l_idx].ve0.xyz);
        pp[l_idx].ve1.w = dot(pp[l_idx].oc_pos, pp[l_idx].ve1.xyz);
        pp[l_idx].ve2.w = dot(pp[l_idx].oc_pos, pp[l_idx].ve2.xyz);
        pp[l_idx].ve3.w = dot(pp[l_idx].oc_pos, pp[l_idx].ve3.xyz);

        v0 = MVP * v0;
        v1 = MVP * v1;
        v2 = MVP * v2;
        v3 = MVP * v3;

        v0 /= v0.w;
        v1 /= v1.w;
        v2 /= v2.w;
        v3 /= v3.w;

        mins = v0.xy;
        maxs = v0.xy;
        mins = min(mins, v1.xy);
        maxs = max(maxs, v1.xy);
        mins = min(mins, v2.xy);
        maxs = max(maxs, v2.xy);
        mins = min(mins, v3.xy);
        maxs = max(maxs, v3.xy);

        vec2 factor = 0.5f * viewAttr.zw;
        v0.xy = factor * (v0.xy + 1.0f);
        v1.xy = factor * (v1.xy + 1.0f);
        v2.xy = factor * (v2.xy + 1.0f);
        v3.xy = factor * (v3.xy + 1.0f);

        vec2 vw = (v0 - v1).xy;
        vec2 vh = (v2 - v3).xy;

        vec4 projPos = MVP * vec4(pp[l_idx].objPos + pp[l_idx].rad * (camDir), 1.0f);
        projPos = projPos / projPos.w;

        projPos.xy = (mins + maxs) * 0.5f;

        pp[l_idx].vb = factor * (mins.xy + 1.0f);

        float l = max(length(vw), length(vh));

        gl_MeshVerticesNV[l_idx].gl_Position = projPos;
        gl_MeshVerticesNV[l_idx].gl_PointSize = l;

        pp[l_idx].l = 1.0f / l;

        gl_PrimitiveIndicesNV[l_idx] = l_idx;
    }
    gl_PrimitiveCountNV = min(num_points - gl_WorkGroupID.x * gl_WorkGroupSize.x, gl_WorkGroupSize.x);
}
