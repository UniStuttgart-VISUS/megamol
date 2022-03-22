#version 450

#include "common/common.inc.glsl"

layout(location = 0) out vec4 fragColor;
layout(location = 1) out float selectColor;
layout(early_fragment_tests) in;

in vec4 color;

void main() {
    fragColor = color;
    // Hack to store selection in a second color attachment, which is maybe faster than using GL_RG32F instead of GL_R32F for the first attachment.
    selectColor = color.g;
}
