#version 130

#include "core/primitives/vertex_attributes.glsl"
#include "core/primitives/screenspace.glsl"

void main() {
    vec4 inPos = mvp * vec4(inPosition, 1.0);
    inPos /= inPos.w;
    gl_Position = inPos;
    color = inColor;
    texcoord = inTexture;
    center = toScreenSpace(inPos.xy, viewport);
    vec4 radPos = mvp * vec4(inAttributes.xyz, 1.0);
    radPos /= radPos.w;
    vec2 radPixel = toScreenSpace(radPos.xy, viewport);
    radius = length(radPixel - center);
    gl_PointSize = radius * 2.0;
}
