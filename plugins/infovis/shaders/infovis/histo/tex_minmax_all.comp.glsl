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
uniform uint texHeight;

layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint col = gl_GlobalInvocationID.x;

    if (col >= numCols) {
        return;
    }

    for (int y = 0; y < texHeight; y++) {
        minValues[col] = min(minValues[col], minValues[(y + 1) * numCols + col]);
        maxValues[col] = max(maxValues[col], maxValues[(y + 1) * numCols + col]);
    }
}
