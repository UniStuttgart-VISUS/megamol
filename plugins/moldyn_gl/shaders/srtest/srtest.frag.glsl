#version 450

flat in vec3 objPos;
flat in float rad;
flat in float sqrRad;
flat in vec4 pointColor;
flat in vec3 oc_pos;
flat in float c;

#include "srtest_ubo.glsl"

layout(location = 0) out vec4 outColor;
layout(depth_greater) out float gl_FragDepth;

#include "lightdirectional.glsl"

#include "srtest_intersection.glsl"

#include "srtest_depth.glsl"

void main(void) {
    vec4 new_pos;
    vec3 normal;
    vec3 ray;
    float t;
    intersection(objPos, sqrRad, oc_pos, c, rad, new_pos, normal, ray, t);

    outColor = vec4(LocalLighting(ray, normal, lightDir, pointColor.rgb), pointColor.a);

    gl_FragDepth = depth(t);
}
