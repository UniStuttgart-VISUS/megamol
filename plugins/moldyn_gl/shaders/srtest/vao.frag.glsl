#version 450

uniform vec4 viewAttr;

uniform mat4 MVPinv;
uniform mat4 MVPtransp;

uniform vec3 camPos;

uniform vec3 lightDir;

uniform float near;
uniform float far;

flat in vec3 objPos;
flat in float rad;
flat in float sqrRad;
flat in vec4 pointColor;

layout (location = 0) out vec4 outColor;
layout (depth_greater) out float gl_FragDepth;

#include "lightdirectional.glsl"

void main(void) {
    vec3 pos_w = gl_FragCoord.xyz;

    vec3 pos_ndc = vec3(2.0 * (pos_w.xy / viewAttr.zw) - 1.0, (2.0 * pos_w.z) / (far - near) - 1.0);
    vec4 pos_clip = (MVPinv * vec4(pos_ndc, 1.0));
    vec3 pos_obj = pos_clip.xyz / pos_clip.w;

    vec3 ray = normalize(pos_obj - camPos);

    vec3 oc = camPos - objPos;
    float b = dot(ray, oc);
    float c = dot(oc, oc) - sqrRad;

    float d = b * b - c;

    if (d >= 0.0) {
        outColor = pointColor;
        float ta = -b - sqrt(d);
        float tb = -b + sqrt(d);
        float t = min(ta, tb);
        vec4 new_pos = vec4(camPos + t * ray, 1.0);

        vec3 normal = (new_pos.xyz - objPos) / rad;
        outColor = vec4(LocalLighting(ray, normal, lightDir, outColor.rgb), outColor.a);

        float depth = dot(MVPtransp[2], new_pos);
        float depthW = dot(MVPtransp[3], new_pos);
        gl_FragDepth = ((depth / depthW) + 1.0) * 0.5;
    } else {
        // outColor = vec4(1, 0, 0, 1);
        discard;
    }
}
