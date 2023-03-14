#version 450

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

#undef __SRTEST_UPLOAD_NO_SEP__
#define __SRTEST_UPLOAD_COPY_IN__
#include "srtest_ssbo.glsl"

uniform uint num_points;

struct Out_Point {
    vec3 pos;
    uint col;
};

layout(std430, binding = 10) writeonly buffer Point_Out {
    Out_Point in_data[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < num_points) {
        in_data[idx].pos = vec3(xPtr[idx], yPtr[idx], zPtr[idx]);
        in_data[idx].col = packUnorm4x8(vec4(rPtr[idx], gPtr[idx], bPtr[idx], aPtr[idx]));
    }
}
