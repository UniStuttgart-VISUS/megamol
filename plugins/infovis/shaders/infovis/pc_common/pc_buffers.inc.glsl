layout(std430, binding = 0) buffer Data
{
    float data[];
};

layout(std430, binding = 1) buffer Flags
{
    coherent uint flags[];
};

layout(std430, binding = 2) buffer DataMinimum
{
    float dataMinimum[];
};

layout(std430, binding = 3) buffer DataMaximum
{
    float dataMaximum[];
};

layout(std430, binding = 4) buffer AxisIndirection
{
    uint axisIndirection[];
};

struct DimensionFilter
{
    uint dimension;
    float lower;
    float upper;
    uint flags;
};

layout(std430, binding = 5) buffer Filters
{
    DimensionFilter filters[];
};

layout(std430, binding = 6) buffer FragmentMinMax
{
    coherent uint minFragmentCount;
    coherent uint maxFragmentCount;
};
