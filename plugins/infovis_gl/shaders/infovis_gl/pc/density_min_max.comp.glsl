#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

#include "common/common.inc.glsl"

layout(binding = 1) uniform sampler2D fragmentCountTex;
uniform ivec2 resolution = ivec2(0);
uniform uvec2 fragmentCountStepSize = uvec2(16);

void main() {
    if (any(greaterThanEqual(gl_GlobalInvocationID.xy, resolution))) {
        return;
    }

    memoryBarrierBuffer();

    uint thisMin = 4294967295u;
    uint thisMax = 0;

    ivec2 texCoord = ivec2(0);
    texCoord.y = int(gl_GlobalInvocationID.y);
    while (texCoord.y < resolution.y) {
        texCoord.x = int(gl_GlobalInvocationID.x);
        while (texCoord.x < resolution.x) {
            uint count = uint(texelFetch(fragmentCountTex, texCoord, 0).r);
            if (count < thisMin) {
                thisMin = count;
            }
            if (count > thisMax) {
                thisMax = count;
            }
            texCoord.x += int(fragmentCountStepSize.x);
        }
        texCoord.y += int(fragmentCountStepSize.y);
    }

    atomicMin(densityMin, thisMin);
    atomicMax(densityMax, thisMax);
}
