#version 430

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec4 in_orient;
uniform mat4 mvp;
uniform float direction_len;

out vec4 vertColor;

void main(void) {
    vertColor = vec4(1.0, 1.0, 1.0, 1.0);
    vec3 pos = in_pos + vec3(direction_len, 0.0, 0.0);
    gl_Position = mvp * vec4(pos, 1.0);
}
