#version 450

#include "srtest_ext.glsl"

flat in vec4 pointColor;
flat in vec3 oc_pos;
flat in float rad;
flat in float sqrRad;

#include "srtest_ubo.glsl"

layout(location = 0) out vec4 outColor;

#include "lightdirectional.glsl"

#include "srtest_intersection.glsl"

#include "srtest_depth.glsl"

void main() {
    vec3 normal;
    vec3 ray;
    float t;
    gl_FragDepth = gl_FragCoord.z;
    if (intersection_old(oc_pos, sqrRad, rad, normal, ray, t)) {
        outColor = vec4(LocalLighting(ray.xyz, normal, lightDir, pointColor.rgb), pointColor.a);
        gl_FragDepth = depth(t);
    } else {
        //outColor = vec4(1.0f, 174.0f / 256.0f, 201.0f / 256.0f, 1.0f);
        discard;
    }

}
