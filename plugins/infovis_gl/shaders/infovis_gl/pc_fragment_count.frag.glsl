#version 430

#include "core/tflookup.inc.glsl"
#include "core/tfconvenience.inc.glsl"

#include "pc_common/pc_fragment_count_buffers.inc.glsl"
#include "pc_common/pc_fragment_count_uniforms.inc.glsl"

smooth in vec2 texCoord;

layout(location = 0) out vec4 fragColor;

void main() {
    float frags = texture(fragmentCount, texCoord).r;
    float select = texture(selectionFlag, texCoord).r;

    uvec2 globalMinMax = fragmentMinMax[0].xy;

    if (select > 0) {
        fragColor = vec4(1.0, 0.0, 0.0, 1.0);
    } else if (frags > 0) {
        float value = (frags - globalMinMax[0]) / (globalMinMax[1] - globalMinMax[0]);
        if (sqrtDensity == 1) {
            value = sqrt(value);
        }
        value = clamp(value, 0.0, 1.0);
        fragColor = tflookup(value);
    } else {
        discard;
    }
}
