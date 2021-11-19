#version 450

in Point {
    flat vec3 objPos;
    flat float rad;
    flat float sqrRad;
    flat vec4 pointColor;
    flat vec3 oc_pos;
    flat float c;
}
pp;

#include "srtest_ubo.glsl"

layout(location = 0) out vec4 outColor;
layout(depth_greater) out float gl_FragDepth;

#include "lightdirectional.glsl"

#include "srtest_intersection.glsl"

#include "srtest_depth.glsl"

void main() {
    vec4 new_pos;
    vec3 normal;
    vec3 ray;
    float t;
    intersection(pp.objPos, pp.sqrRad, pp.oc_pos, pp.c, pp.rad, new_pos, normal, ray, t);

    outColor = vec4(LocalLighting(ray, normal, lightDir, pp.pointColor.rgb), pp.pointColor.a);

    gl_FragDepth = depth(t);
}
