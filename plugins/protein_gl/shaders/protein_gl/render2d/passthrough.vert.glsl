#version 460

layout (location = 0) in vec2 vert_position;
layout (location = 1) in vec3 vert_color;

uniform bool use_per_vertex_color = false;
uniform vec3 global_color = vec3(0.5, 0.5, 0.5);
uniform float global_alpha = 1.0;
uniform mat4 mvp;

out vec4 color;

void main(void) {
    gl_Position = mvp * vec4(vert_position, 0, 1);
    color = vec4(vert_color, global_alpha);
    if(!use_per_vertex_color){
        color = vec4(global_color, global_alpha);
    }
}