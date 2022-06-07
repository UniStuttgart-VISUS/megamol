#version 460

//#define BASE_IDX gl_VertexID / 6
//#define INV_IDX gl_VertexID % 3
//#define BUMP_IDX gl_VertexID % 6 / 3

uniform vec4 globalCol;
uniform float globalRad;

uniform bool useGlobalCol;
uniform bool useGlobalRad;

flat out vec3 objPos;
flat out float rad;
flat out float sqrRad;
flat out vec4 pointColor;
flat out vec3 oc_pos;

//uniform int offset;

#include "srtest_ubo.glsl"

#ifdef __SRTEST_VAO__
#include "srtest_vao.glsl"
#elif defined(__SRTEST_SSBO__)
#include "srtest_ssbo.glsl"
#endif

#include "srtest_touchplane.glsl"

#include "srtest_frustum.glsl"

#ifdef __SRTEST_MUZIC__
layout(std430, binding = 10) readonly buffer OffsetBuf {
    uint offset_cmd[];
};
#endif

void main() {
#ifdef __SRTEST_MUZIC__
    int offset = int(offset_cmd[gl_DrawID]);

    /*int base_idx = gl_InstanceID;
    int inv_idx = gl_VertexID;*/
    /*int base_idx = gl_VertexID / 4;
    int inv_idx = gl_VertexID % 4;*/
    int base_idx = BASE_IDX + offset;
#else
    int base_idx = BASE_IDX;
#endif
    int inv_idx = INV_IDX;
    int bump_idx = BUMP_IDX;

    access_data(base_idx, objPos, pointColor, rad);

    oc_pos = objPos - camPos;
    sqrRad = rad * rad;

    /*float dd = dot(oc_pos, oc_pos);

    float s = (sqrRad) / (dd);

    float vi = rad / sqrt(1.0f - s);

    vec3 vr = normalize(cross(oc_pos, camUp)) * vi;
    vec3 vu = normalize(cross(oc_pos, vr)) * vi;*/

    //#if BUMP_IDX == 0
    /*mat4 v = mat4(vec4(objPos - vr - vu, 1.0f), vec4(objPos + vr - vu, 1.0f), vec4(objPos + vr + vu, 1.0f),
        vec4(objPos - vr + vu, 1.0f));*/
    //#else
    /*mat4 v = mat4(vec4(objPos - vr - vu, 1.0f), vec4(objPos + vr - vu, 1.0f), vec4(objPos - vr + vu, 1.0f),
        vec4(objPos + vr + vu, 1.0f));*/
    /*mat4 v = mat4(vec4(objPos - vr + vu, 1.0f), vec4(objPos - vr - vu, 1.0f),
        vec4(objPos + vr + vu, 1.0f), vec4(objPos + vr - vu, 1.0f));*/
    //#endif

    mat4 v;
#ifdef __SRTEST_QUAD__
    touchplane_old(objPos, rad, oc_pos, v);
    //touchplane(objPos, rad, oc_pos, v);
#else
    touchplane_old_v2(objPos, rad, oc_pos, v);
#endif

    vec4 pos = v[inv_idx + bump_idx];

    //pos /= pos.w;

    //vec4 projPos = MVP * vec4(objPos + rad * (camDir), 1.0f);

    //pos.z = projPos.z / projPos.w;

    /*if (isOutside(oc_pos, rad)) {
        pos.xyz = vec3(0);
    }*/

    //gl_Position = vec4(pos.xy, projPos.zw);
    gl_Position = pos;
}
