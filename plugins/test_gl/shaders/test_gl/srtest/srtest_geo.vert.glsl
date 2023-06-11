#version 450

uniform vec4 globalCol;
uniform float globalRad;

uniform bool useGlobalCol;
uniform bool useGlobalRad;

out VPoint {
    flat float rad;
    flat float sqrRad;
    flat vec4  pointColor;
    flat vec3  oc_pos;
}
v_pp;

#include "srtest_ubo.glsl"

#ifdef __SRTEST_VAO__
#include "srtest_vao.glsl"
#elif defined(__SRTEST_SSBO__)
#include "srtest_ssbo.glsl"
#endif

void main() {
    vec3 objPos;
    access_data(gl_VertexID, objPos, v_pp.pointColor, v_pp.rad);

    v_pp.oc_pos = objPos - camPos;
    v_pp.sqrRad = v_pp.rad * v_pp.rad;

    gl_Position = vec4(objPos, 1.0f);
}
