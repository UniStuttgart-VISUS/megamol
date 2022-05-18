layout(std430, binding = 0) buffer Histogram
{
    int histogram[];
};

layout(std430, binding = 1) buffer SelectedHistogram
{
    int selectedHistogram[];
};

layout(std430, binding = 2) buffer Minimums
{
    float minimums[];
};

layout(std430, binding = 3) buffer Maximums
{
    float maximums[];
};

uniform uint numBins = 0;
uniform uint numComponents = 0;
