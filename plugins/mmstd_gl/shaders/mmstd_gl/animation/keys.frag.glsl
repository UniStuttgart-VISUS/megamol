#version 430

in vec4 vertColor;
layout(location = 0) out vec4 col;

void main(void) {
    col = vertColor;
    // "always on top"
    gl_FragDepth = 0.0;
}
