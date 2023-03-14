#version 450

//#pragma optionNV(fastmath off)
//#pragma optionNV(fastprecision off)

#extension GL_ARB_conservative_depth : enable

in Point {
    //flat vec3 objPos;
    flat float rad;
    flat float sqrRad;
    flat vec4 pointColor;
    flat vec3 oc_pos;
    //flat float c;
    //flat vec3 new_camPos;
}
pp;

#include "srtest_ubo.glsl"

layout(location = 0) out vec4 outColor;
layout(depth_greater) out float gl_FragDepth;

#include "lightdirectional.glsl"

#include "srtest_intersection.glsl"

#include "srtest_depth.glsl"

//bool intersection_old2(vec3 oc_pos, float sqrRad, float rad, out vec3 normal, out vec3 ray, out float t) {
//    // transform fragment coordinates from window coordinates to view coordinates.
//    /*vec4 coord =
//        vec4(2.0f * (gl_FragCoord.xy / viewAttr.zw) - 1.0f, (2.0f * gl_FragCoord.z) / (far - near) - 1.0f, 1.0f);*/
//    vec4 coord = gl_FragCoord * vec4(2.0f / viewAttr.z, 2.0f / viewAttr.w, 2.0, 0.0) + vec4(-1.0, -1.0, -1.0, 1.0);
//
//    // transform fragment coordinates from view coordinates to object coordinates.
//    coord = MVPinv * coord;
//    coord /= coord.w;
//
//    ray = normalize(coord.xyz - camPos);
//
//
//
//    // calculate the geometry-ray-intersection
//    float b = dot(oc_pos, ray); // projected length of the cam-sphere-vector onto the ray
//    vec3 temp = oc_pos - b * ray;
//    float l2 = dot(temp, temp);
//    if (l2 > sqrRad)
//        return false;
//
//    float td = sqrt(sqrRad - l2);
//    t = b - td;
//
//    //vec3 temp = b * ray - oc_pos;
//    //float delta = sqrRad - dot(temp, temp); // Raytracing Gem Magic (http://www.realtimerendering.com/raytracinggems/)
//
//    //if (delta < 0.0f)
//    //    return false;
//
//    //float c = dot(oc_pos, oc_pos) - sqrRad;
//
//    //float s = b < 0.0f ? -1.0f : 1.0f;
//    //float q = b + s * sqrt(delta);
//    //t = min(c / q, q);
//
//    vec3 sphereintersection = t * ray - oc_pos; // intersection point
//    normal = (sphereintersection) / rad;
//
//    return true;
//}

void main() {
    /*vec4 new_pos;
    vec3 normal;
    vec3 ray;
    float t;
    intersection(pp.objPos, pp.sqrRad, pp.oc_pos, pp.c, pp.rad, new_pos, normal, ray, t);*/

    /*vec4 pos_ndc =
        vec4(2.0f * (gl_FragCoord.xy / viewAttr.zw) - 1.0f, (2.0f * gl_FragCoord.z) / (far - near) - 1.0f, 1.0f);
    vec4 pos_clip = MVPinv * pos_ndc;
    vec3 pos_obj = pos_clip.xyz / pos_clip.w;

    vec3 ray = normalize(pos_obj - camPos);

    float tf = dot(pp.oc_pos, ray);
    vec3 tt = tf * ray - pp.oc_pos;
    float delta = pp.sqrRad - dot(tt, tt);
    if (delta < 0.0f)
        discard;

    float tb = sqrt(delta);
    float t = tf - tb;

    vec4 new_pos = vec4(camPos + t * ray, 1.0f);

    vec3 normal = (new_pos.xyz - pp.objPos) / pp.rad;

    outColor = vec4(LocalLighting(ray, normal, lightDir, pp.pointColor.rgb), pp.pointColor.a);

    gl_FragDepth = depth(t);*/

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
    }
}
