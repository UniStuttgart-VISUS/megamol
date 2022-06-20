#version 460

layout (location = 0) in vec3 vert_position;
layout (location = 1) in vec3 vert_normal;
layout (location = 2) in vec3 vert_color;

out vec3 normal;
out vec4 color;

uniform mat4 mvp;
uniform float alpha = 1.0;

void main() {
    gl_Position = mvp * vec4(vert_position, 1.0);
    normal = normalize(vert_normal);
    color = vec4(vert_color, alpha);
}
