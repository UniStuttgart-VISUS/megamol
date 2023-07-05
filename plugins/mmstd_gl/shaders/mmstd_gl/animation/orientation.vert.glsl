#version 430

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec4 in_orient;

out vec4 vs_orient;

void main(void) {
    vs_orient = in_orient;
    gl_Position = vec4(in_pos, 1.0);
}
