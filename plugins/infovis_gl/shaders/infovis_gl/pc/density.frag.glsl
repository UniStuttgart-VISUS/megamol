#version 450

#include "common/common.inc.glsl"
#include "mmstd_gl/common/tflookup.inc.glsl"
#include "mmstd_gl/common/tfconvenience.inc.glsl"

layout(binding = 1) uniform sampler2D fragmentCountTex;
layout(binding = 2) uniform sampler2D selectionFlagTex;

uniform bool normalizeDensity = true;
uniform bool sqrtDensity = false;

smooth in vec2 texCoord;

layout(location = 0) out vec4 fragColor;

void main() {
    float fragmentCount = texture(fragmentCountTex, texCoord).r;
    float selected = texture(selectionFlagTex, texCoord).r;

    if (selected > 0) {
        fragColor = vec4(1.0, 0.0, 0.0, 1.0);
    } else if (fragmentCount > 0) {
        if (normalizeDensity) {
            fragmentCount = (fragmentCount - densityMin) / (densityMax - densityMin);
            if (sqrtDensity) {
                fragmentCount = sqrt(fragmentCount);
            }
            fragmentCount = clamp(fragmentCount, 0.0, 1.0);
        }
        fragColor = tflookup(fragmentCount);
    } else {
        discard;
    }
}
