#version 450

uniform vec4 indicatorColor = vec4(0.0f, 0.0f, 1.0f, 1.0f);

in vec2 circleCoord;

out vec4 fragColor;

void main() {
    const float dist = length(circleCoord);

    if (dist < 1.0f) {
        fragColor = indicatorColor;
    } else {
        discard;
    }
}
