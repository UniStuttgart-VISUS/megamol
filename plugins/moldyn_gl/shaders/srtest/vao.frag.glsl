#version 450

flat in vec3 objPos;
flat in float rad;
flat in float sqrRad;
flat in vec4 pointColor;
flat in vec3 oc_pos;
flat in float dot_oc_pos;

#include "srtest_ubo.glsl"

layout(location = 0) out vec4 outColor;
layout(depth_greater) out float gl_FragDepth;

#include "lightdirectional.glsl"

void main(void) {
    vec4 pos_ndc = vec4(2.0f * (gl_FragCoord.xy / viewAttr.zw) - 1.0f, (2.0f * gl_FragCoord.z) / (far - near) - 1.0f, 1.0f);
    vec4 pos_clip = MVPinv * pos_ndc;
    vec3 pos_obj = pos_clip.xyz / pos_clip.w;

    vec3 ray = normalize(pos_obj - camPos);

    float b = dot(-oc_pos, ray);
    vec3 temp = oc_pos + b * ray;
    float delta = sqrRad - dot(temp, temp);

    if (delta < 0.0f)
        discard;

    float c = dot_oc_pos - sqrRad;
    float sign = b >= 0.0f ? 1.0f : -1.0f;
    float q = b + sign * sqrt(delta);

    float t = min(c / q, q);

    vec4 new_pos = vec4(camPos + t * ray, 1.0f);

    vec3 normal = (new_pos.xyz - objPos) / rad;
    outColor = vec4(LocalLighting(ray, normal, lightDir, pointColor.rgb), pointColor.a);

    float depth = dot(MVPtransp[2], new_pos);
    float depthW = dot(MVPtransp[3], new_pos);
    gl_FragDepth = ((depth / depthW) + 1.0f) * 0.5f;
}
