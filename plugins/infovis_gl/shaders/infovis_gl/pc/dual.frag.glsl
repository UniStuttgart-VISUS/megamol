#version 450

#include "common/common.inc.glsl"

in vec2 uvCoords;
out vec4 fragOut;
in vec4 color;
layout (binding=7, r32ui) uniform uimage2DArray imgRead;

void main() {
    fragOut = vec4(1.0);
}
