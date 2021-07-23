#version 430

#include "core/bitflags.inc.glsl"

layout(std430, binding = 1) buffer Minimums
{
    float minimums[];
};

layout(std430, binding = 2) buffer Maximums
{
    float maximums[];
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

int local_histo[256];
int local_selected[256];

uniform sampler2D anytex;

layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;

void main() {
    ivec2 ts = textureSize(anytex, 0);
    uint col = gl_LocalInvocationID.x;
    uint scanline = gl_WorkGroupID.x;
    if (scanline >= ts.y || col >= colCount) {
        return;
    }

    // TODO find minmax first and exchange

    float localmin = 3.402823466e+38;
    float localmax = -localmin;

    for (int x = 0; x < ts.x; ++x) {
        uint lin = scanline * ts.x + x;
        if (bitflag_isVisible(lin)) {
            float val = texelFetch(anytex, ivec2(x, scanline), 0)[col];
            localmin = min(localmin, val);
            localmax = max(localmax, val);
        }
    }

    // memoryBarrier();
    localmin = minimums[col];
    localmax = maximums[col];

    // TODO better idea?
    for (int x = 0; x < binCount; ++x) {
        local_histo[x] = 0;
        local_selected[x] = 0;
    }
    for (int x = 0; x < ts.x; ++x) {
        uint lin = scanline * ts.x + x;
        if (bitflag_isVisible(lin)) {
            float val = (texelFetch(anytex, ivec2(x, scanline), 0)[col] - localmin) / (localmax - localmin);
            int bin_idx = clamp(int(val * binCount), 0, int(binCount) - 1);
            local_histo[bin_idx] = local_histo[bin_idx] + 1;
            if (bitflag_isVisibleSelected(flags[lin])) {
                local_selected[bin_idx] = local_selected[bin_idx] + 1;
            }
        }
    }

    for (int x = 0; x < binCount; ++x) {
        atomicAdd(histogram[x * colCount + col], local_histo[x]);
        atomicAdd(selectedHistogram[x * colCount + col], local_selected[x]);
    }

}
