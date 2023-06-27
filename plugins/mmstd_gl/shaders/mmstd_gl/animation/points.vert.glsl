#version 430
#extension GL_ARB_shader_storage_buffer_object : require

#include "point-data.inc.glsl"

uniform mat4 mvp;

out vec4 vertColor;

void main(void) {
    vec4 pos = vec4(points[gl_VertexID].x, points[gl_VertexID].y, points[gl_VertexID].z, 1.0);
    vertColor = unpackUnorm4x8(points[gl_VertexID].col);
    gl_Position = mvp * pos;

    // throw away background / empty pixels
    gl_ClipDistance[0] = vertColor.a > 0.0 ? 1.0 : -1.0;
}
