#version version 330

#include "common_defines_btf.glsl"
#include "lightdirectional.glsl"

layout(location = 0) out vec4 frag_color;
layout(location = 1) out vec4 value_color;

in vec4 colorval;
in float value;

void main(void) {
    frag_color = colorval;
    value_color = vec4(value, 0.0, 0.0, 0.0);
}
