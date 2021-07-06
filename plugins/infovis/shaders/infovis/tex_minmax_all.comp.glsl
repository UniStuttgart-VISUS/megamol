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

layout(std430, binding = 8) buffer LineMinimums
{
    float lineMinimums[];
};

layout(std430, binding = 9) buffer LineMaximums
{
    float lineMaximums[];
};

layout(std430, binding = 3) buffer Flags
{
    coherent uint flags[];
};

uniform uint colCount = 0;

uniform sampler2D anytex;

layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;

void main() {
    ivec2 ts = textureSize(anytex, 0);
    uint col = gl_LocalInvocationID.x;
    uint scanline = gl_WorkGroupID.x;
    if (scanline >= ts.y || col >= colCount) {
        return;
    }

    // minmax over all scanlines

    for (int y = 0; y < ts.y; ++y) {
        minimums[col] = min(minimums[col], lineMinimums[y * colCount + col]);
        maximums[col] = max(maximums[col], lineMaximums[y * colCount + col]);
    }
}
