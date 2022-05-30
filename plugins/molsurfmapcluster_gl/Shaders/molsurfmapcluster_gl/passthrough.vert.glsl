#version 460

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec3 in_color;
layout(location = 2) in int in_id;

uniform mat4 mvp;

out vec4 color;
flat out int id;

void main() {
    gl_Position = mvp * vec4(in_position, 0, 1);
    color = vec4(in_color, 1);
    id = in_id;
}