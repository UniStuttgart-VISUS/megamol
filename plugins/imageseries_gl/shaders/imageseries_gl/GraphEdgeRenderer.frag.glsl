#version 430

layout(location = 0) in float edgeWeight;

out vec4 fragColor;

void main() {
    fragColor = vec4(0.0, 0.0, 0.0, min(edgeWeight / 100.0, 1.0));
}
