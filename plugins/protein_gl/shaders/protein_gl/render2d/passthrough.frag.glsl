#version 460

layout(location = 0) out vec4 color_out;

in vec3 color;

void main(void) {
    color_out = vec4(color, 1);
}