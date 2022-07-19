#version 430

#include "mmstd_gl/common/tflookup.inc.glsl"
#include "mmstd_gl/common/tfconvenience.inc.glsl"
#include "mmstd_gl/flags/bitflags.inc.glsl"
#include "common/mapping.inc.glsl"

uniform mat4 modelViewProjection;

in vec2 position;
in float normalizedValue;

out float vsValue;
out vec4 vsValueColor;

void main() {
    vsValue = normalizedValue;
    vsValueColor = tflookup(normalizedValue);
    gl_Position = modelViewProjection * vec4(position, 0.0, 1.0);
}
