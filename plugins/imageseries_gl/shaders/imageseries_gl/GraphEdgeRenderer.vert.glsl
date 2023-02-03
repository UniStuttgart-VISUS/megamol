#version 430

uniform mat4 matrix;

layout(location = 0) in vec2 pos;

void main() {
    gl_Position = matrix * vec4(pos, 0.0, 1.0);
}
