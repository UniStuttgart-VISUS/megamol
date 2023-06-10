#version 460

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

    int base_idx = BASE_IDX + offset;
#else
    int base_idx = BASE_IDX;
#endif
    int inv_idx = INV_IDX;
    int bump_idx = BUMP_IDX;

    vec3 objPos;
    access_data(base_idx, objPos, pointColor, rad);

    oc_pos = objPos - camPos;
    sqrRad = rad * rad;

    mat4 v;
#ifdef __SRTEST_QUAD__
#ifdef __SRTEST_CAM_ALIGNED__
    touchplane(objPos, rad, oc_pos, v);
#else
    touchplane_old(objPos, rad, oc_pos, v);
#endif
    //touchplane(objPos, rad, oc_pos, v);
#else
    touchplane_old_v2(objPos, rad, oc_pos, v);
#endif

    vec4 pos = v[inv_idx + bump_idx];
    
    gl_Position = pos;
}
