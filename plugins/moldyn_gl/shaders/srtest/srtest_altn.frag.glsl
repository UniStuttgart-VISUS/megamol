#version 450

#include "srtest_ext.glsl"

flat in vec3 objPos;
flat in float rad;
flat in float sqrRad;
flat in vec4 pointColor;
flat in vec3 oc_pos;
flat in vec4 ve0;
flat in vec4 ve1;
flat in vec4 ve2;
flat in vec4 ve3;
flat in vec2 vb;
flat in float l;

//flat in vec3 tt0;
//flat in vec3 tt1;
//flat in vec3 tt2;
//flat in vec3 tt3;

#include "srtest_ubo.glsl"

layout(location = 0) out vec4 outColor;
//layout(depth_greater) out float gl_FragDepth;

#include "lightdirectional.glsl"

#include "srtest_intersection.glsl"

#include "srtest_depth.glsl"

void main(void) {
    /*vec4 new_pos;
    vec3 normal;
    vec3 ray;
    float t;
    intersection(objPos, sqrRad, oc_pos, c, rad, new_pos, normal, ray, t);*/

    vec2 factor = (gl_FragCoord.xy - vb) * l;

    vec4 v_bot = mix(ve0, ve1, factor.x);
    vec4 v_top = mix(ve3, ve2, factor.x);
    vec4 ray = mix(v_top, v_bot, factor.y);

    /*vec3 tt_bot = mix(tt0, tt1, factor.x);
    vec3 tt_top = mix(tt3, tt2, factor.x);
    vec3 tt = mix(tt_top, tt_bot, factor.y);*/

    float tf = ray.w;

    //float tf = dot(oc_pos, ray.xyz);
    vec3 tt = tf * ray.xyz - oc_pos;
    float delta = sqrRad - dot(tt, tt);
    if (delta < 0.0f)
        discard;

    float tb = sqrt(delta);
    float t = tf - tb;

    vec4 new_pos = vec4(camPos + t * ray.xyz, 1.0f);

    vec3 normal = (new_pos.xyz - objPos) / rad;

    outColor = vec4(LocalLighting(ray.xyz, normal, lightDir, pointColor.rgb), pointColor.a);
    /*outColor = vec4(factor, 0, 1);
    outColor = vec4(ve0, 1);*/
    //outColor = vec4(0.5f*(ray+1.0f), 1);

    gl_FragDepth = depth(t);
}
