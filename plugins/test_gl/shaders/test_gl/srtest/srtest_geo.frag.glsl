#version 450

#include "srtest_ext.glsl"

layout(location = 0) out vec4 outColor;

#include "srtest_ubo.glsl"

in Point {
    flat float rad;
    flat float sqrRad;
    flat vec4 pointColor;
    flat vec3 oc_pos;
}
g_pp;

#include "lightdirectional.glsl"

#include "srtest_intersection.glsl"

#include "srtest_depth.glsl"

void main() {
    vec3 normal;
    vec3 ray;
    float t;
    gl_FragDepth = gl_FragCoord.z;
    if (intersection_old(g_pp.oc_pos, g_pp.sqrRad, g_pp.rad, normal, ray, t)) {
        outColor = vec4(LocalLighting(ray.xyz, normal, lightDir, g_pp.pointColor.rgb), g_pp.pointColor.a);
        gl_FragDepth = depth(t);
    } else {
        //outColor = vec4(1.0f, 174.0f / 256.0f, 201.0f / 256.0f, 1.0f);
        discard;
    }

}
