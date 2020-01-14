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
    float histogram[];
};

layout(std430, binding = 5) buffer SelectedHistogram
{
    float selectedHistogram[];
};

uniform uint binCount = 0;
uniform uint colCount = 0;
uniform uint rowCount = 0;

layout(local_size_x = 1) in;

void main() {
    // TODO this is just a copy of the old CPU code, at the moment this is sequential shader code and pretty useless
    if (gl_GlobalInvocationID.x == 0) {
        // buffer memory is uninitialized
        for (int i = 0; i < colCount * binCount; i++) {
            histogram[i] = 0.0;
            selectedHistogram[i] = 0.0;
        }

        for (uint c = 0; c < colCount; c++) {
            for (uint r = 0; r < rowCount; r++) {
                if (bitflag_isVisible(flags[r])) {
                    float val = (floatData[r * colCount + c] - minimums[c]) / (maximums[c] - minimums[c]);
                    int bin_idx = clamp(int(val * binCount), 0, int(binCount) - 1);
                    histogram[bin_idx * colCount + c] += 1.0;
                    if (bitflag_isVisibleSelected(flags[r])) {
                        selectedHistogram[bin_idx * colCount + c] += 1.0;
                    }
                }
            }
        }
    }
}
