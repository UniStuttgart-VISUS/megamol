#version 430

#include "pc_common/pc_extensions.inc.glsl"
#include "pc_common/pc_useLineStrip.inc.glsl"
#include "pc_common/pc_buffers.inc.glsl"
#include "pc_common/pc_uniforms.inc.glsl"
#include "pc_common/pc_common.inc.glsl"
//#include "::pc_item_pick::uniforms"

smooth out vec2 circleCoord;

void main()
{
    vec2 vertices[6] =
    {
    // b_l, b_r, t_r
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0)
    // t_r, t_l, b_l
    , vec2(1.0, 1.0), vec2(-1.0, 1.0), vec2(-1.0, -1.0)
    };

    circleCoord = vertices[gl_VertexID];

    vec4 vertex = vec4(mouse + pickRadius * circleCoord, pc_item_defaultDepth, 1.0);

    gl_Position = projection * modelView * vertex;
}
