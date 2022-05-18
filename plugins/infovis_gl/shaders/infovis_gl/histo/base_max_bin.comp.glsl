#version 430

#include "common.inc.glsl"

layout(std430, binding = 4) buffer MaxBinValue
{
    int maxBinValue[];
};

layout(local_size_x = 1024) in;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint bufSize = numComponents * numBins;

    // divide size of histogram buffer by number of local threads to distribute work evenly.
    uint localSize = (bufSize > 0 ? bufSize - 1 : 0) / 1024 + 1;

    uint localIdxStart = idx * localSize;
    uint localIdxEnd = (idx + 1) * localSize;

    int maxVal = 0;
    for (uint i = localIdxStart; i < localIdxEnd && i < bufSize; i++) {
        if (histogram[i] > maxVal) {
            maxVal = histogram[i];
        }
    }

    atomicMax(maxBinValue[0], maxVal);
}
