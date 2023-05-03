#version 430

in vec4 vertColor;
layout(location = 0) out vec4 col;

void main(void) {
    col = vertColor;
}
