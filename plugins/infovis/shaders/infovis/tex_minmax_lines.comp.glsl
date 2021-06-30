#version 430

#include "core/bitflags.inc.glsl"

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

    // minmax per scanline

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
    lineMinimums[scanline * colCount + col] = localmin;
    lineMaximums[scanline * colCount + col] = localmax;
}
