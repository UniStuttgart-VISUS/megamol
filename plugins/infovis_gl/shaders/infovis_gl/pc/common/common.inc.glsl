/**
 * Common buffers and uniforms set within useProgramAndBindCommon().
 */

// Structs
struct Range {
    float min;
    float max;
};

// Buffers
layout(std430, binding = 0) buffer Data {
    float data[];
};

layout(std430, binding = 1) coherent buffer Flags {
    uint flags[];
};

layout(std430, binding = 2) buffer DataRanges {
    Range dataRange[];
};

layout(std430, binding = 3) buffer AxisIndirection {
    uint axisIndirection[];
};

layout(std430, binding = 4) buffer Filters {
    Range filters[];
};

layout(std430, binding = 5) coherent buffer DensityMinMax {
    uint densityMin;
    uint densityMax;
};

// Uniforms
uniform mat4 projMx = mat4(1.0f);
uniform mat4 viewMx = mat4(1.0f);

uniform uint dimensionCount = 0;
uniform uint itemCount = 0;

uniform vec2 margin = vec2(0.0f, 0.0f);
uniform float axisDistance = 0.0f;
uniform float axisHeight = 0.0f;

// Shader constants
const float pc_defaultDepth = 0.0f;

// Functions
float pc_normalizeDimension(float val, uint dimensionIdx) {
    return (val - dataRange[dimensionIdx].min) / (dataRange[dimensionIdx].max - dataRange[dimensionIdx].min);
}

float pc_dataValue(uint itemIdx, uint dimensionIdx) {
    return data[itemIdx * dimensionCount + dimensionIdx];
}

float pc_dataValueNormalized(uint itemIdx, uint dimensionIdx) {
    return pc_normalizeDimension(pc_dataValue(itemIdx, dimensionIdx), dimensionIdx);
}

vec4 pc_axisVertex(uint axisIdx, float factor) {
    return vec4(
        margin.x + axisDistance * axisIdx,
        margin.y + axisHeight * factor,
        pc_defaultDepth,
        1.0f);
}

vec4 pc_itemVertex(uint itemIdx, uint axisIdx) {
    return pc_axisVertex(axisIdx, pc_dataValueNormalized(itemIdx, axisIndirection[axisIdx]));
}
