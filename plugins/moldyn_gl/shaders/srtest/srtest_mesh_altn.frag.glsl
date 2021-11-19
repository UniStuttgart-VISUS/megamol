#version 450

in Point {
    flat vec4 pointColor;
    flat vec3 objPos;
    vec3 ray;
    vec3 tt;
    flat float rad;
    flat float sqrRad;
    float tf;
}
pp;

#include "srtest_ubo.glsl"

layout(location = 0) out vec4 outColor;
layout(depth_greater) out float gl_FragDepth;

#include "lightdirectional.glsl"

#include "srtest_intersection.glsl"

void main() {
    float delta = pp.sqrRad - dot(pp.tt, pp.tt);
    if (delta < 0.0f)
        discard;

    float tb = sqrt(delta);
    float t = pp.tf - tb;

    vec4 new_pos = vec4(camPos + t * pp.ray, 1.0f);

    vec3 normal = (new_pos.xyz - pp.objPos) / pp.rad;

    outColor = vec4(LocalLighting(pp.ray, normal, lightDir, pp.pointColor.rgb), pp.pointColor.a);

    float depth = dot(MVPtransp[2], new_pos);
    float depthW = dot(MVPtransp[3], new_pos);
    gl_FragDepth = ((depth / depthW) + 1.0f) * 0.5f;
}
