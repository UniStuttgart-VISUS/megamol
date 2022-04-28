#version 450

#include "common/common.inc.glsl"
#include "mmstd_gl/common/quad_vertices.inc.glsl"

uniform vec2 mouse = vec2(0.0f, 0.0f);
uniform float pickRadius = 0.1f;

smooth out vec2 circleCoord;

void main() {
    vec2 pos = quadVertexPosition() * 2.0f - 1.0f;

    circleCoord = pos;

    vec4 vertex = vec4(mouse + pickRadius * pos, pc_defaultDepth, 1.0f);
    gl_Position = projMx * viewMx * vertex;
}
