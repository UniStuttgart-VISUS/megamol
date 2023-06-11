#version 450

uniform vec4 globalCol;
uniform float globalRad;

uniform bool useGlobalCol;
uniform bool useGlobalRad;

flat out float rad;
flat out float sqrRad;
flat out vec4 pointColor;
flat out vec3 oc_pos;

#include "srtest_ubo.glsl"

#ifdef __SRTEST_VAO__
#include "srtest_vao.glsl"
#elif defined(__SRTEST_SSBO__)
#include "srtest_ssbo.glsl"
#elif defined(__SRTEST_TEX__)
#include "srtest_tex.glsl"
#endif

#include "srtest_touchplane.glsl"

void main(void) {
    vec3 objPos;
    access_data(gl_VertexID, objPos, pointColor, rad);

    sqrRad = rad * rad;

    oc_pos = objPos - camPos;

    vec4 projPos;
    float l;
    touchplane_old(objPos, rad, oc_pos, projPos, l);

    gl_PointSize = l;
    gl_Position = projPos;
}
