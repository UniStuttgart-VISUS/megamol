layout(std430, binding = 0) buffer FloatData
{
    float floatData[];
};

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
uniform uint rowCount = 0;

layout(local_size_x = 1024) in;

void main() {
    uint col = gl_GlobalInvocationID.x;
    if (col >= colCount) {
        return;
    }

    for (uint r = 0; r < rowCount; r++) {
        if (bitflag_isVisible(flags[r])) {
            float val = (floatData[r * colCount + col] - minimums[col]) / (maximums[col] - minimums[col]);
            int bin_idx = clamp(int(val * binCount), 0, int(binCount) - 1);
            histogram[bin_idx * colCount + col] += 1;
            if (bitflag_isVisibleSelected(flags[r])) {
                selectedHistogram[bin_idx * colCount + col] += 1;
            }
        }
    }

    int maxVal = 0;
    for (uint b = 0; b < binCount; b++) {
        if (histogram[b * colCount + col] > maxVal) {
            maxVal = histogram[b * colCount + col];
        }
    }

    atomicMax(maxBinValue[0], maxVal);
}
