#version 430

#include "core/bitflags.inc.glsl"

layout(std430, binding = 0) buffer FloatData
{
    float floatData[];
};

layout(std430, binding = 3) buffer Flags
{
    coherent uint flags[];
};

layout(std430, binding = 4) buffer Histogram
{
    int histogram[];
};

layout(std430, binding = 5) buffer SelectedHistogram
{
    int selectedHistogram[];
};

layout(std430, binding = 6) buffer MaxBinValue
{
    int maxBinValue[];
};

uniform uint binCount = 0;
uniform uint colCount = 0;
//uniform uint rowCount = 0;

int local_histo[255];
int local_selected[255];

uniform sampler2D anytex;

//TODO all wrong, make a local histogram per scanline or something and then add shit together.

layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;

void main() {
    ivec2 ts = textureSize(anytex, 0);
    uint col = gl_LocalInvocationID.x;
    uint scanline = gl_WorkGroupID.x;
    if (scanline >= ts.y || col >= colCount) {
        return;
    }
    for (int x = 0; x < binCount; ++x) {
        local_histo[x] = 0;
        local_selected[x] = 0;
    }
    for (int x = 0; x < ts.x; ++x) {
        uint lin = scanline * ts.x + x;
        if (bitflag_isVisible(lin)) {
            float val = texelFetch(anytex, ivec2(x, scanline), 0)[col];
            int bin_idx = clamp(int(val * binCount), 0, int(binCount) - 1);
            //atomicAdd(histogram[bin_idx * colCount + col], 1);
            local_histo[bin_idx] = local_histo[bin_idx] + 1;
            if (bitflag_isVisibleSelected(flags[lin])) {
                //atomicAdd(selectedHistogram[bin_idx * colCount + col], 1);
                local_selected[bin_idx] = local_selected[bin_idx] + 1;
            }
        }
    }
    //for (uint r = 0; r < rowCount; r++) {
    //    if (bitflag_isVisible(flags[r])) {
    //        float val = texelfetch //floatData[r * colCount + col];
    //        int bin_idx = clamp(int(val * binCount), 0, int(binCount) - 1);
    //        histogram[bin_idx * colCount + col] += 1;
    //        if (bitflag_isVisibleSelected(flags[r])) {
    //            selectedHistogram[bin_idx * colCount + col] += 1;
    //        }
    //    }
    //}

    for (int x = 0; x < binCount; ++x) {
        atomicAdd(histogram[x * colCount + col], local_histo[x]);
        atomicAdd(selectedHistogram[x * colCount + col], local_selected[x]);
    }

    //memoryBarrier();

    //if (gl_LocalInvocationID.x == 0 && gl_LocalInvocationID.y == 0 && gl_LocalInvocationID.z == 0) {
    //    int maxVal = 0;
    //    for (uint b = 0; b < binCount; b++) {
    //        for (uint c = 0; c < colCount; c++) {
    //            if (histogram[b * colCount + c] > maxVal) {
    //                maxVal = histogram[b * colCount + c];
    //            }
    //        }
    //    }

    //    //atomicMax(maxBinValue[0], maxVal);
    //    maxBinValue[0] = maxVal;
    //}
    //maxBinValue[0] = 50;
}
