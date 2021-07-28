#version 430

#include "common.inc.glsl"

int local_histo[256];

uniform sampler2D tex;

layout(local_size_x = 256, local_size_y = 4, local_size_z = 1) in;

void main() {
    const uint texY = gl_GlobalInvocationID.x;
    const uint col = gl_GlobalInvocationID.y;

    const ivec2 texSize = textureSize(tex, 0);

    if (texY >= texSize.y || col > numCols) {
        return;
    }

    // clamp bins to 256 because of the tmp array size
    // TODO draw histogram will not know this and just displays empty bins above!
    const uint overwriteNumBin = min(numBins, 256);

    for (int i = 0; i < overwriteNumBin; i++) {
        local_histo[i] = 0;
    }

    const float minVal = minimums[col];
    const float maxVal = maximums[col];

    for (int x = 0; x < texSize.x; x++) {
        float val = (texelFetch(tex, ivec2(x, texY), 0)[col] - minVal) / (maxVal - minVal);
        int bin_idx = clamp(int(val * overwriteNumBin), 0, int(overwriteNumBin) - 1);
        local_histo[bin_idx] += 1;
    }

    for (int i = 0; i < overwriteNumBin; i++) {
        atomicAdd(histogram[i * numCols + col], local_histo[i]);
    }
}
