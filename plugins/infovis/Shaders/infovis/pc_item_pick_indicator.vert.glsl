#version 430

#include <snippet name="::pc::extensions" />
#include <snippet name="::pc::useLineStrip" />
#include <snippet name="::pc::buffers" />
#include <snippet name="::pc::uniforms" />
#include <snippet name="::pc::common" />
#include <snippet name="::pc_item_pick::uniforms" />

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
