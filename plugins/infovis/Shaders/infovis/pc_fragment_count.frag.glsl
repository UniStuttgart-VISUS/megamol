#version 430

#include "pc_common/pc_fragment_count_buffers.inc.glsl"
#include "pc_common/pc_fragment_count_uniforms.inc.glsl"

smooth in vec2 texCoord;

layout(location = 0) out vec4 fragColor;

void main()
{
    vec4 frags = texture(fragmentCount, texCoord) - clearColor;

    uvec2 globalMinMax = fragmentMinMax[0].xy;

    if (frags.r >= 0) {
        float value = (frags.r - globalMinMax[0]) / (globalMinMax[1] - globalMinMax[0]);
        if (sqrtDensity == 1) {
            value = sqrt(value);
        }
        value = clamp(value, 0.0, 1.0);
        fragColor = texture(transferFunction, value);
    } else {
        fragColor = clearColor;
    }
}
