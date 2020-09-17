#extension GL_ARB_explicit_attrib_location : enable

layout(location=0) in vec3 inPos;
layout(location=1) in vec4 inColor;

out vec4 color;

uniform mat4 mvp;

void main() {
    color = inColor;
    gl_Position = mvp * vec4(inPos, 1.0);
}