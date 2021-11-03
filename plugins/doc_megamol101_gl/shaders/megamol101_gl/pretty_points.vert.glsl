#version 400

layout(location = 0) in vec4 vPosition;
layout(location = 1) in vec4 vColor;

uniform float scalingFactor = 1.0;
uniform mat4 mvp;
out vec4 color;
out float myRadius;

void main() {
    vec4 pos = vec4(vPosition.xyz, 1.0);
    gl_Position = pos;
    color = vColor;
    myRadius = vPosition.w * scalingFactor;
}
