#version 430

layout(location = 0) in vec2 pos;

void main() {
    gl_Position = vec4(pos + vec2(0.5), 0.0, 1.0);
}
