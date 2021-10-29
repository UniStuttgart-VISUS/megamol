#version 430

layout(std430, binding = 0) buffer MinValues
{
    float minValues[];
};

layout(std430, binding = 1) buffer MaxValues
{
    float maxValues[];
};

uniform uint numComponents;
uniform uint texHeight;

layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint component = gl_GlobalInvocationID.x;

    if (component >= numComponents) {
        return;
    }

    for (int y = 0; y < texHeight; y++) {
        minValues[component] = min(minValues[component], minValues[(y + 1) * numComponents + component]);
        maxValues[component] = max(maxValues[component], maxValues[(y + 1) * numComponents + component]);
    }
}
