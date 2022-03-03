#version 430

#include "pc_common/pc_buffers.inc.glsl"
#include "pc_common/pc_uniforms.inc.glsl"
#include "pc_common/pc_common.inc.glsl"

uniform int width;
uniform int height;
uniform float thickness;

void main() {
    vec2 from = mouseReleased;
    vec2 to = mousePressed;

    vec2 dir = (projection * modelView * vec4(to,0,0) - projection * modelView * vec4(from,0,0)).xy;
    vec2 offset = normalize(vec2(-dir.y * height, dir.x * width));
    offset = vec2(offset.x / width, offset.y / height);
    offset = thickness * offset;

    int side = gl_VertexID / 2 - gl_VertexID/3;
    vec4 vertex = vec4((1 - (gl_VertexID % 2)) * from + (gl_VertexID % 2) * to, pc_item_defaultDepth, 1.0);

    gl_Position = (projection * modelView * vertex) + vec4(1 * offset.xy, 0, 0) + side * vec4(2* -offset, 0, 0);
}
