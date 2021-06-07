#version 430

#include "core/tflookup.inc.glsl"
#include "core/tfconvenience.inc.glsl"

uniform vec4 selectionColor;

in float binColor;
in float selection;

void main(void) {
    if (selection <= 1.0) {
        gl_FragColor = selectionColor;
    } else {
        gl_FragColor = tflookup(binColor);
    }
}
