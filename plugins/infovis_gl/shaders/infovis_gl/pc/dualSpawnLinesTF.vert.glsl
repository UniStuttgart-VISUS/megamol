#version 450

#include "common/common.inc.glsl"
#include "mmstd_gl/common/quad_vertices.inc.glsl"

smooth out vec2 texCoord;

void main() {
    vec2 pos = quadVertexPosition();

    texCoord = pos;
    gl_Position = vec4(pos * 2.0f - 1.0f, pc_defaultDepth, 1.0f);
}
