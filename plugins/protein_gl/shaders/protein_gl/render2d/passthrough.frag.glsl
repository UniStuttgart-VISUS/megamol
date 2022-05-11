#version 460

layout(location = 0) out vec4 color_out;

in vec4 color;

void main(void) {
    color_out = color;
}