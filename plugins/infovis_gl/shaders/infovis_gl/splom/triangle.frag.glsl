#version 430

#include "mmstd_gl/common/tflookup.inc.glsl"
#include "mmstd_gl/common/tfconvenience.inc.glsl"
#include "mmstd_gl/flags/bitflags.inc.glsl"
#include "common/mapping.inc.glsl"

in float vsValue;
in vec4 vsValueColor;

layout(location = 0) out vec4 fsColor;

void main() {
    fsColor = toScreen(vsValue, vsValueColor, 1.0);
}
