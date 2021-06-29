#version 430

layout(std430, binding = 4) buffer Histogram
{
    int histogram[];
};

layout(std430, binding = 6) buffer MaxBinValue
{
    int maxBinValue[];
};

uniform uint binCount = 0;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() {
    maxBinValue[0] = 0;
    for (int x = 0; x < binCount; ++x) {
        maxBinValue[0] = max(maxBinValue[0], histogram[x]);
    }
}
