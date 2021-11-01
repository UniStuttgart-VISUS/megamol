#version 430

#include "pc_common/pc_extensions.inc.glsl"
#include "pc_common/pc_useLineStrip.inc.glsl"
#include "pc_common/pc_buffers.inc.glsl"
#include "pc_common/pc_uniforms.inc.glsl"
#include "pc_common/pc_common.inc.glsl"
//#include "::pc_item_stroke::uniforms"

uniform int width;
uniform int height;
uniform float thickness;

void main()
{
    #if 0
    float left = abscissae[0] * scaling.x;
    float bottom = 0.0f;
    float right = abscissae[dimensionCount - 1] * scaling.x;
    float top = scaling.y;

    //vec2 extent = vec2(right - left, bottom - top);
    vec2 extent = vec2(1920,1080);
    #endif
    #if 1
    vec2 from = mouseReleased;
    vec2 to = mousePressed;
    #else
    vec2 from = vec2(left,bottom);
    vec2 to = mouseReleased*vec2(right,top);
    #endif

    vec2 dir = (projection * modelView * vec4(to,0,0) - projection * modelView * vec4(from,0,0)).xy;
    vec2 offset = normalize(vec2(-dir.y * height, dir.x * width));
    offset = vec2(offset.x / width, offset.y / height);
    offset = thickness * offset;

    int side = gl_VertexID / 2 - gl_VertexID/3;
    vec4 vertex = vec4((1 - (gl_VertexID % 2)) * from + (gl_VertexID % 2) * to, pc_item_defaultDepth, 1.0);

    gl_Position = (projection * modelView * vertex) + vec4(1 * offset.xy, 0, 0) + side * vec4(2* -offset, 0, 0);
}
