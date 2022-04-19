#version 430

uniform mat4 matrix;

layout(location = 0) in vec2 pos;

out vec2 texCoord;

void main() {
    texCoord = vec2(pos.x, 1.0 - pos.y);
    gl_Position = matrix * vec4(pos - vec2(0.5), -10.0, 1.0);
}
