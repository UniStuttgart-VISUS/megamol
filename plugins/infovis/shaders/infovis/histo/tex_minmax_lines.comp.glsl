#version 430

layout(std430, binding = 0) buffer MinValues
{
    float minValues[];
};

layout(std430, binding = 1) buffer MaxValues
{
    float maxValues[];
};

uniform uint numCols;
uniform sampler2D tex;

layout(local_size_x = 256, local_size_y = 4, local_size_z = 1) in;

void main() {
    const uint texY = gl_GlobalInvocationID.x;
    const uint col = gl_GlobalInvocationID.y;

    const ivec2 texSize = textureSize(tex, 0);

    if (texY >= texSize.y || col > numCols) {
        return;
    }

    // minValues and maxValues are initialized from CPU
    float localMin = minValues[col];
    float localMax = maxValues[col];

    for (uint x = 0; x < texSize.x; x++) {
        float val = texelFetch(tex, ivec2(x, texY), 0)[col];
        localMin = min(localMin, val);
        localMax = max(localMax, val);
    }

    minValues[(texY + 1) * numCols + col] = localMin;
    maxValues[(texY + 1) * numCols + col] = localMax;
}
