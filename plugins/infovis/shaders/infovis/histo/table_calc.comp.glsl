#version 430

#include "core/bitflags.inc.glsl"
#include "common.inc.glsl"

layout(std430, binding = 4) buffer FloatData
{
    float floatData[];
};

layout(std430, binding = 5) buffer Flags
{
    coherent uint flags[];
};

uniform uint numRows = 0;

layout(local_size_x = 1024) in;

void main() {
    uint col = gl_GlobalInvocationID.x;
    if (col >= numCols) {
        return;
    }

    for (uint r = 0; r < numRows; r++) {
        if (bitflag_isVisible(flags[r])) {
            float val = (floatData[r * numCols + col] - minimums[col]) / (maximums[col] - minimums[col]);
            int bin_idx = clamp(int(val * numBins), 0, int(numBins) - 1);
            histogram[bin_idx * numCols + col] += 1;
            if (bitflag_isVisibleSelected(flags[r])) {
                selectedHistogram[bin_idx * numCols + col] += 1;
            }
        }
    }
}
