#version 450

uniform vec4 indicatorColor = vec4(0.0f, 0.0f, 1.0f, 1.0f);

out vec4 fragColor;

void main() {
    fragColor = indicatorColor;
}
