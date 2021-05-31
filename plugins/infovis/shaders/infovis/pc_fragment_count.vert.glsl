#version 430

#include "pc_common/pc_fragment_count_buffers.inc.glsl"
#include "pc_common/pc_fragment_count_uniforms.inc.glsl"

uniform vec2 bottomLeft = vec2(-1.0);
uniform vec2 topRight = vec2(1.0);
uniform float depth = 0.0;

smooth out vec2 texCoord;

void main()
{
    const vec2 vertices[6] =
    {
    // b_l, b_r, t_r
    bottomLeft, vec2(topRight.x, bottomLeft.y), topRight
    // t_r, t_l, b_l
    , topRight, vec2(bottomLeft.x, topRight.y), bottomLeft
    };

    const vec2 texCoords[6] =
    {
    // b_l, b_r, t_r
    vec2(0.0), vec2(1.0, 0.0), vec2(1.0)
    // t_r, t_l, b_l
    , vec2(1.0), vec2(0.0, 1.0), vec2(0.0)
    };

    texCoord = texCoords[gl_VertexID];

    vec4 vertex = vec4(vertices[gl_VertexID], depth, 1.0);

    gl_Position = /*projection * modelView */ vertex;
}
