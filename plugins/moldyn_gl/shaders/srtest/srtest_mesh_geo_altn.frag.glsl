#version 450

in Point {
    flat vec4 pointColor;
    flat vec3 objPos;
    vec4 ray;
    flat vec3 oc_pos;
    flat float rad;
    flat float sqrRad;
}
pp;

#include "srtest_ubo.glsl"

layout(location = 0) out vec4 outColor;
layout(depth_greater) out float gl_FragDepth;

#include "lightdirectional.glsl"

#include "srtest_intersection.glsl"

#include "srtest_depth.glsl"

void main() {
    gl_FragDepth = gl_FragCoord.z;
    // float tf = dot(pp.oc_pos, pp.ray);
    vec3 tt = pp.ray.w * pp.ray.xyz - pp.oc_pos;
    float delta = pp.sqrRad - dot(tt, tt);
    if (delta < 0.0f)
        discard;

    /*float tb = sqrt(delta);
    float t = pp.ray.w - tb;*/

    float c = dot(pp.oc_pos, pp.oc_pos) - pp.sqrRad;

    float s = pp.ray.w < 0.0f ? -1.0f : 1.0f;
    float q = pp.ray.w + s * sqrt(delta);
    float t = min(c / q, q);

    vec4 new_pos = vec4(camPos + t * pp.ray.xyz, 1.0f);

    vec3 normal = (new_pos.xyz - pp.objPos) / pp.rad;

    outColor = vec4(LocalLighting(pp.ray.xyz, normal, lightDir, pp.pointColor.rgb), pp.pointColor.a);
    // outColor = vec4(0.5f * (pp.ray + 1.0f), 1);

    /*if (delta < 0.0f)
        outColor = vec4(1.0f, 174.0f / 256.0f, 201.0f / 256.0f, 1.0f);*/

    gl_FragDepth = depth(t);
}
