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
flat out float c;

#include "srtest_ubo.glsl"

#ifdef __SRTEST_VAO__
#include "srtest_vao.glsl"
#elif defined(__SRTEST_SSBO__)
#include "srtest_ssbo.glsl"
#endif

#include "srtest_touchplane.glsl"

void main(void) {
    access_data(gl_VertexID, objPos, pointColor, rad);

    oc_pos = camPos - objPos;
    sqrRad = rad * rad;
    c = dot(oc_pos, oc_pos) - sqrRad;

    vec4 projPos;
    float l;
    touchplane(objPos, rad, projPos, l);

    gl_PointSize = l;

    gl_Position = projPos;
}
