#version 450

uniform vec4 globalCol;
uniform float globalRad;

uniform bool useGlobalCol;
uniform bool useGlobalRad;

flat out vec3 objPos;
flat out float rad;
flat out float sqrRad;
flat out vec4 pointColor;
flat out vec3 oc_pos;

#include "srtest_ubo.glsl"

#ifdef __SRTEST_VAO__
#include "srtest_vao.glsl"
#elif defined(__SRTEST_SSBO__)
#include "srtest_ssbo.glsl"
#endif

void main() {
    int base_idx = gl_VertexID / 4;
    int inv_idx = gl_VertexID % 4;

    access_data(base_idx, objPos, pointColor, rad);

    oc_pos = objPos - camPos;
    sqrRad = rad * rad;

    float dd = dot(oc_pos, oc_pos);

    float s = (sqrRad) / (dd);

    float vi = rad / sqrt(1.0f - s);

    vec3 vr = normalize(cross(oc_pos, camUp)) * vi;
    vec3 vu = normalize(cross(oc_pos, vr)) * vi;

    mat4 v = mat4(vec4(objPos - vr - vu, 1.0f), vec4(objPos + vr - vu, 1.0f), vec4(objPos + vr + vu, 1.0f),
        vec4(objPos - vr + vu, 1.0f));

    vec4 pos = MVP * v[inv_idx];
    pos /= pos.w;

    vec4 projPos = MVP * vec4(objPos + rad * (camDir), 1.0f);
    projPos = projPos / projPos.w;

    gl_Position = vec4(pos.xy, projPos.z, 1.0f);
}
