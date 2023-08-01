#version 430

layout(location = 0) in vec2 pos;
layout(location = 1) in float weightIn;

layout(location = 0) out float weight;

void main() {
    weight = weightIn;
    gl_Position = vec4(pos + vec2(0.5), 0.0, 1.0);
}
