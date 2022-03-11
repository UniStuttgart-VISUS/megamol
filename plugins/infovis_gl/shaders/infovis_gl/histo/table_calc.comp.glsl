#version 430

#include "core/bitflags.inc.glsl"
#include "common.inc.glsl"

layout(std430, binding = 4) buffer FloatData
{
    float floatData[];
};

layout(std430, binding = 5) coherent buffer Flags
{
    uint flags[];
};

uniform uint numRows = 0;

layout(local_size_x = 1024) in;

void main() {
    uint component = gl_GlobalInvocationID.x;
    if (component >= numComponents) {
        return;
    }

    for (uint r = 0; r < numRows; r++) {
        if (bitflag_isVisible(flags[r])) {
            float val = (floatData[r * numComponents + component] - minimums[component]) / (maximums[component] - minimums[component]);
            int bin_idx = clamp(int(val * numBins), 0, int(numBins) - 1);
            histogram[bin_idx * numComponents + component] += 1;
            if (bitflag_isVisibleSelected(flags[r])) {
                selectedHistogram[bin_idx * numComponents + component] += 1;
            }
        }
    }
}
