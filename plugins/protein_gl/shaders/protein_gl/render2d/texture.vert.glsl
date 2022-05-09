#version 460

layout (location = 0) in vec2 vert_position;
layout (location = 1) in vec2 vert_texcoord;

uniform mat4 mvp;

void main(void) {
    gl_Position = mvp * vec4(vert_position, 0, 1);
}