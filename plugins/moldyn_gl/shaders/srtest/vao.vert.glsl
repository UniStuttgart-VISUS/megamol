#version 450

uniform vec4 globalCol;
uniform float globalRad;

uniform bool useGlobalCol;
uniform bool useGlobalRad;

flat out vec3 objPos;
flat out float rad;
flat out float sqrRad;
flat out vec4 pointColor;

#include "srtest_ubo.glsl"

layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec4 inColor;

void main(void) {
    if (useGlobalRad) {
        rad = globalRad;
    } else {
        rad = inPosition.w;
    }

    if (useGlobalCol) {
        pointColor = globalCol;
    } else {
        pointColor = inColor;
    }

    objPos = inPosition.xyz;

    vec2 mins, maxs;

    vec3 di = objPos - camPos;
    float dd = dot(di, di);

    sqrRad = rad * rad;

    float s = (sqrRad) / (dd);

    float v = rad / sqrt(1 - s);

    vec3 vr = normalize(cross(di, camUp)) * v;
    vec3 vu = normalize(cross(di, vr)) * v;

    vec4 v1 = MVP * vec4(objPos + vr, 1.0);
    vec4 v2 = MVP * vec4(objPos - vr, 1.0);
    vec4 v3 = MVP * vec4(objPos + vu, 1.0);
    vec4 v4 = MVP * vec4(objPos - vu, 1.0);

    v1 /= v1.w;
    v2 /= v2.w;
    v3 /= v3.w;
    v4 /= v4.w;

    mins = v1.xy;
    maxs = v1.xy;
    mins = min(mins, v2.xy);
    maxs = max(maxs, v2.xy);
    mins = min(mins, v3.xy);
    maxs = max(maxs, v3.xy);
    mins = min(mins, v4.xy);
    maxs = max(maxs, v4.xy);

    v1.xy = 0.5 * v1.xy * viewAttr.zw + 0.5 * viewAttr.zw;
    v2.xy = 0.5 * v2.xy * viewAttr.zw + 0.5 * viewAttr.zw;
    v3.xy = 0.5 * v3.xy * viewAttr.zw + 0.5 * viewAttr.zw;
    v4.xy = 0.5 * v4.xy * viewAttr.zw + 0.5 * viewAttr.zw;

    vec2 vw = (v1 - v2).xy;
    vec2 vh = (v3 - v4).xy;

    vec4 projPos = MVP * vec4(objPos + rad * (camDir), 1.0);
    projPos = projPos / projPos.w;

    gl_PointSize = max(length(vw), length(vh));

    gl_Position = vec4((mins + maxs) * 0.5, projPos.z, 1.0);
}
