#version 430

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec3 in_normal;
layout(location = 3) in vec2 in_texcoord;

out vec3 pass_pos;
out vec3 pass_normal;
out vec3 pass_color;
out vec2 pass_texcoord;

uniform mat4 mvp;

void main() {
    pass_pos = in_position;
    pass_normal = in_normal;
    pass_color = in_color;
    pass_texcoord = in_texcoord;
    gl_Position = mvp * vec4(in_position, 1.0);
}
