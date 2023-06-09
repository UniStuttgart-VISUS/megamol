#version 450

#extension GL_ARB_gpu_shader_fp64 : enable

#include "srtest_ext.glsl"

#pragma optionNV(fastmath off)
#pragma optionNV(fastprecision off)

in Point {
    flat vec3 objPos;
    flat float rad;
    flat float sqrRad;
    flat vec4 pointColor;
    flat vec3 oc_pos;
    flat float c;
    flat in vec4 ve0;
    flat in vec4 ve1;
    flat in vec4 ve2;
    flat in vec4 ve3;
    flat in vec2 vb;
    flat in float l;
}
pp;

#include "srtest_ubo.glsl"

layout(location = 0) out vec4 outColor;
//layout(depth_greater) out float gl_FragDepth;

#include "lightdirectional.glsl"

#include "srtest_intersection.glsl"

#include "srtest_depth.glsl"

vec3 slerp(vec3 from, vec3 to, float t) {
    float theta = acos(dot(from, to));
    float sin_theta = sin(theta);
    float a = sin((1 - t) * theta) / sin_theta;
    float b = sin(t * theta) / sin_theta;

    return from * a + to * b;
}

void main() {
    /*vec4 new_pos;
    vec3 normal;
    vec3 ray;
    float t;
    intersection(pp.objPos, pp.sqrRad, pp.oc_pos, pp.c, pp.rad, new_pos, normal, ray, t);*/

    vec2 factor = (gl_FragCoord.xy - pp.vb) * pp.l;

    vec4 v_bot = mix(pp.ve0, pp.ve1, factor.x);
    vec4 v_top = mix(pp.ve3, pp.ve2, factor.x);
    vec4 ray = mix(v_top, v_bot, factor.y);

    /*vec3 ray_bot = slerp(pp.ve0.xyz, pp.ve1.xyz, factor.x);
    vec3 ray_top = slerp(pp.ve3.xyz, pp.ve2.xyz, factor.x);
    vec3 r_ray = slerp(ray_top, ray_bot, factor.y);*/

    dvec3 ray_bot = mix(dvec3(pp.ve0.xyz), dvec3(pp.ve1.xyz), double(factor.x));
    dvec3 ray_top = mix(dvec3(pp.ve3.xyz), dvec3(pp.ve2.xyz), double(factor.x));
    vec3 r_ray = vec3(mix(dvec3(ray_top), dvec3(ray_bot), double(factor.y)));

    /*vec3 tt_bot = mix(tt0, tt1, factor.x);
    vec3 tt_top = mix(tt3, tt2, factor.x);
    vec3 tt = mix(tt_top, tt_bot, factor.y);*/

    //float tf = ray.w;

    float tf = dot(pp.oc_pos, r_ray.xyz);
    vec3 tt = tf * r_ray.xyz - pp.oc_pos;
    float delta = pp.sqrRad - dot(tt, tt);
    if (delta < 0.0f)
        discard;

    /*float tb = sqrt(delta);
    float t = tf - tb;*/

    float c = dot(pp.oc_pos, pp.oc_pos) - pp.sqrRad;

    float s = tf < 0.0f ? -1.0f : 1.0f;
    float q = tf + s * sqrt(delta);
    float t = min(c / q, q);

    vec4 new_pos = vec4(camPos + t * r_ray.xyz, 1.0f);

    vec3 normal = (new_pos.xyz - pp.objPos) / pp.rad;

    outColor = vec4(LocalLighting(r_ray.xyz, normal, lightDir, pp.pointColor.rgb), pp.pointColor.a);

    gl_FragDepth = depth(t);
}
