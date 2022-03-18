#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

#include "common/common.inc.glsl"

layout(binding = 1) uniform sampler2D fragmentCountTex;
uniform ivec2 resolution = ivec2(0);
uniform uint blockSize = 16u;

void main() {
    // Check pixels in a blockSize * blockSize block within the fbo texture.
    // Calculate start and end corner positions in tex coords of the current invocation.
    uvec2 texCoordsStart = gl_GlobalInvocationID.xy * uvec2(blockSize);
    if (texCoordsStart.x >= resolution.x || texCoordsStart.y >= resolution.y) {
        return;
    }
    uvec2 texCoordsEnd = (gl_GlobalInvocationID.xy + 1) * uvec2(blockSize);
    texCoordsEnd = min(texCoordsEnd, uvec2(resolution));

    uint localMin = 4294967295u;
    uint localMax = 0;

    for (uint y = texCoordsStart.y; y < texCoordsEnd.y; y++) {
        for (uint x = texCoordsStart.x; x < texCoordsEnd.x; x++) {
            uint count = uint(texelFetch(fragmentCountTex, ivec2(x, y), 0).r);
            if (count < localMin) {
                localMin = count;
            }
            if (count > localMax) {
                localMax = count;
            }
        }
    }

    atomicMin(densityMin, localMin);
    atomicMax(densityMax, localMax);
}
