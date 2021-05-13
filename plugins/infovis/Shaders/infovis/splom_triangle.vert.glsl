#version 430

#include "core/tflookup.inc.glsl"
#include "core/tfconvenience.inc.glsl"
#include "core/bitflags.inc.glsl"
#include "splom_common/splom_mapping.inc.glsl"

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
