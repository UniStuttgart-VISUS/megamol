#version 450

uniform vec4 globalCol;
uniform float globalRad;

uniform bool useGlobalCol;
uniform bool useGlobalRad;

flat out vec3 objPos;
flat out float rad;
flat out float sqrRad;
flat out vec4 pointColor;
flat out vec3 oc_pos;
flat out float c;
flat out vec4 ve0;
flat out vec4 ve1;
flat out vec4 ve2;
flat out vec4 ve3;
flat out vec2 vb;
flat out float l;

//flat out vec3 tt0;
//flat out vec3 tt1;
//flat out vec3 tt2;
//flat out vec3 tt3;

#include "srtest_ubo.glsl"

#ifdef __SRTEST_VAO__
#include "srtest_vao.glsl"
#elif defined(__SRTEST_SSBO__)
#include "srtest_ssbo.glsl"
#endif

#include "srtest_touchplane.glsl"

void main(void) {
    access_data(gl_VertexID, objPos, pointColor, rad);

    // oc_pos = camPos - objPos;
    oc_pos = objPos - camPos;
    sqrRad = rad * rad;

    /*vec4 projPos;
    float l;
    touchplane(objPos, rad, projPos, l);*/


    vec2 mins, maxs;

    float dd = dot(oc_pos, oc_pos);

    float s = (sqrRad) / (dd);

    float vi = rad / sqrt(1.0f - s);

    vec3 vr = normalize(cross(oc_pos, camUp)) * vi;
    vec3 vu = normalize(cross(oc_pos, vr)) * vi;

    vec4 v0 = vec4(objPos + vr, 1.0f);
    vec4 v1 = vec4(objPos - vr, 1.0f);
    vec4 v2 = vec4(objPos + vu, 1.0f);
    vec4 v3 = vec4(objPos - vu, 1.0f);

    ve0 = vec4(normalize((v1.xyz - vu) - camPos), 0.0);
    ve1 = vec4(normalize((v0.xyz - vu) - camPos), 0.0);
    ve2 = vec4(normalize((v0.xyz + vu) - camPos), 0.0);
    ve3 = vec4(normalize((v1.xyz + vu) - camPos), 0.0);

    ve0.w = dot(oc_pos, ve0.xyz);
    ve1.w = dot(oc_pos, ve1.xyz);
    ve2.w = dot(oc_pos, ve2.xyz);
    ve3.w = dot(oc_pos, ve3.xyz);

    //tt0 = ve0.w * ve0.xyz - oc_pos;
    //tt1 = ve1.w * ve1.xyz - oc_pos;
    //tt2 = ve2.w * ve2.xyz - oc_pos;
    //tt3 = ve3.w * ve3.xyz - oc_pos;

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

    vec4 projPos = MVP * vec4(objPos + rad * (camDir), 1.0f);
    projPos = projPos / projPos.w;

    projPos.xy = (mins + maxs) * 0.5f;

    vb = factor * (mins.xy + 1.0f);

    l = max(length(vw), length(vh));

    gl_PointSize = l;

    l = 1.0f / l;

    gl_Position = projPos;
}
