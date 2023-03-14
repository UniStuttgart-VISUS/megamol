#version 450

#extension GL_ARB_conservative_depth: enable

//flat in vec3 objPos;
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
    gl_FragDepth = gl_FragCoord.z;
    if (intersection_old(oc_pos, sqrRad, rad, normal, ray, t)) {
        outColor = vec4(LocalLighting(ray.xyz, normal, lightDir, pointColor.rgb), pointColor.a);
        gl_FragDepth = depth(t);
        //outColor = vec4(ray,1.0f);
    } else {
        //outColor = vec4(1.0f, 174.0f / 256.0f, 201.0f / 256.0f, 1.0f);
        discard;
    }

}
