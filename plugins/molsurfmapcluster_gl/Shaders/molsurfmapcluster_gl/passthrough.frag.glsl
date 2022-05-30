#version 460

layout(location = 0) out vec4 out_color;
layout(location = 1) out int out_id;

in vec4 color;
flat in int id;

void main() {
    out_color = color;
    out_id = id + 1; // we increment the id so that 0 can be the default value
}