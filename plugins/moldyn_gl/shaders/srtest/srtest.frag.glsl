#version 450

flat in vec3 objPos;
flat in float rad;
flat in float sqrRad;
flat in vec4 pointColor;
flat in vec3 oc_pos;

#include "srtest_ubo.glsl"

layout(location = 0) out vec4 outColor;
layout(depth_greater) out float gl_FragDepth;

#include "lightdirectional.glsl"

#include "srtest_intersection.glsl"

#include "srtest_depth.glsl"

void main(void) {
    vec3 normal;
    vec3 ray;
    float t;
    if (intersection_old(oc_pos, sqrRad, rad, normal, ray, t)) {
        outColor = vec4(LocalLighting(ray.xyz, normal, lightDir, pointColor.rgb), pointColor.a);
        //outColor = vec4(ray,1.0f);
    } else {
        //outColor = vec4(1.0f, 174.0f / 256.0f, 201.0f / 256.0f, 1.0f);
        discard;
    }

    gl_FragDepth = depth(t);
}
