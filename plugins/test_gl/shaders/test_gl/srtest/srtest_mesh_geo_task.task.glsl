#version 450

#extension GL_NV_mesh_shader : enable
#extension GL_NV_shader_thread_group : enable
#extension GL_NV_gpu_shader5 : enable

//#define WARP 32

layout(local_size_x = WARP) in;

taskNV out Task {
    uint baseID;
    uint8_t subIDs[WARP];
}
OUT;

#include "srtest_ubo.glsl"

uniform uint num_points;

uniform vec4 globalCol;
uniform float globalRad;

uniform bool useGlobalCol;
uniform bool useGlobalRad;

#include "srtest_ssbo.glsl"

#include "srtest_frustum.glsl"

void main() {
    uint g_idx = gl_GlobalInvocationID.x;
    bool render = false;

    if (g_idx < num_points) {
        vec3 objPos;
        float rad;
        vec4 pointColor;
        access_data(g_idx, objPos, pointColor, rad);

        vec3 oc_pos = objPos - camPos;

        render = !isOutside(oc_pos, rad);
    }

    uint warp_bitfield = ballotThreadNV(render);
    uint task_count = bitCount(warp_bitfield);

    if (gl_LocalInvocationID.x == 0) {
        OUT.baseID = gl_WorkGroupID.x * WARP;
        gl_TaskCountNV = task_count;
    }

    uint task_out_index = bitCount(warp_bitfield & gl_ThreadLtMaskNV);
    if (render) {
        OUT.subIDs[task_out_index] = uint8_t(gl_LocalInvocationID.x);
    }
}
