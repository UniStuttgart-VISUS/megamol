#version 400

layout(location = 0) in vec4 vPosition;
layout(location = 1) in vec4 vColor;

uniform mat4 mvp;
out vec4 color;
out float radius;

void main() {
    vec4 pos = vec4(vPosition.xyz, 1.0);
    radius = vPosition.w;
    color = vColor;
    gl_Position = mvp * pos;
}
