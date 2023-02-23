#version 430

uniform vec4 axesColor = vec4(1.0f);

layout(location = 0) out vec4 col;

void main(void) {
    col = axesColor;
}
