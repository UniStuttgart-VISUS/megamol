#version 130

#include "core/primitives/vertex_attributes.glsl"
#include "core/primitives/screenspace.glsl"

void main() {
    vec4 inPos = mvp * vec4(inPosition, 1.0);
    inPos /= inPos.w;
    gl_Position = inPos;
    color = inColor;
    texcoord = inTexture;
    attributes = inAttributes;
}
