#version 450

#extension GL_ARB_conservative_depth : enable

in Point {
    flat vec4 pointColor;
    flat vec3 objPos;
    flat vec3 oc_pos;
    flat float rad;
    flat float sqrRad;
}
pp;

#include "srtest_ubo.glsl"

layout(location = 0) out vec4 outColor;
layout(depth_greater) out float gl_FragDepth;
//layout(depth_any) out float gl_FragDepth;

#include "lightdirectional.glsl"

#include "srtest_intersection.glsl"

#include "srtest_depth.glsl"

void main() {
    //vec4 pos_ndc =
    //    vec4(2.0f * (gl_FragCoord.xy / viewAttr.zw) - 1.0f, (2.0f * gl_FragCoord.z) / (far - near) - 1.0f, 1.0f);
    //vec4 pos_clip = MVPinv * pos_ndc;
    //vec3 pos_obj = pos_clip.xyz / pos_clip.w;

    //vec3 ray = normalize(pos_obj - camPos);

    //float tf = dot(pp.oc_pos, ray);
    //vec3 tt = tf * ray - pp.oc_pos;
    //float delta = pp.sqrRad - dot(tt, tt);
    ///*if (delta < 0.0f)
    //    discard;*/

    //float tb = sqrt(delta);
    //float t = tf - tb;

    //vec4 new_pos = vec4(camPos + t * ray, 1.0f);

    //vec3 normal = (new_pos.xyz - pp.objPos) / pp.rad;

    //outColor = vec4(LocalLighting(ray, normal, lightDir, pp.pointColor.rgb), pp.pointColor.a);
    //// outColor = vec4(0.5f * (pp.ray + 1.0f), 1);

    //if (delta < 0.0f)
    //    outColor = vec4(1.0f, 174.0f / 256.0f, 201.0f / 256.0f, 1.0f);

    //gl_FragDepth = depth(t);

    vec3 normal;
    vec3 ray;
    float t;
    gl_FragDepth = gl_FragCoord.z;
    if (intersection_old(pp.oc_pos, pp.sqrRad, pp.rad, normal, ray, t)) {
        outColor = vec4(LocalLighting(ray.xyz, normal, lightDir, pp.pointColor.rgb), pp.pointColor.a);
        gl_FragDepth = depth(t);
    } else {
        //outColor = vec4(1.0f, 174.0f / 256.0f, 201.0f / 256.0f, 1.0f);
        discard;
        //gl_FragDepth = 1.0f;
    }

}
