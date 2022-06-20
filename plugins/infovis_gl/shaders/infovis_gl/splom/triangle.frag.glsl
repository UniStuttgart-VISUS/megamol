#version 430

#include "core/tflookup.inc.glsl"
#include "core/tfconvenience.inc.glsl"
#include "core/bitflags.inc.glsl"
#include "common/mapping.inc.glsl"

in float vsValue;
in vec4 vsValueColor;

layout(location = 0) out vec4 fsColor;

void main() {
    fsColor = toScreen(vsValue, vsValueColor, 1.0);
}
