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

void main(void) {
    vec4 new_pos;
    vec3 normal;
    vec3 ray;
    intersection(objPos, sqrRad, oc_pos, c, rad, new_pos, normal, ray);

    outColor = vec4(LocalLighting(ray, normal, lightDir, pointColor.rgb), pointColor.a);

    float depth = dot(MVPtransp[2], new_pos);
    float depthW = dot(MVPtransp[3], new_pos);
    gl_FragDepth = ((depth / depthW) + 1.0f) * 0.5f;
    //gl_FragDepth = t;
}
