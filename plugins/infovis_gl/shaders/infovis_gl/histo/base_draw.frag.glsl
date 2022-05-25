#version 430

#include "mmstd_gl/common/tflookup.inc.glsl"
#include "mmstd_gl/common/tfconvenience.inc.glsl"

uniform vec4 selectionColor;

in float binColor;
in float selection;

layout(location = 0) out vec4 col;

void main(void) {
    if (selection <= 1.0) {
        col = selectionColor;
    } else {
        col = tflookup(binColor);
    }
}
