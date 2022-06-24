#version 460

layout (location = 0) in vec2 vert_position;
layout (location = 1) in vec3 vert_color;

uniform bool use_per_vertex_color = false;
uniform vec3 global_color = vec3(0.5, 0.5, 0.5);
uniform mat4 mvp;

flat out vec3 startPos;
out vec3 pos;
out vec3 color;

void main(void) {
    vec4 transpos = mvp * vec4(vert_position, 0, 1);
    gl_Position = transpos;
    pos = transpos.xyz / transpos.w;
    startPos = pos;
    color = vert_color;
    if(!use_per_vertex_color){
        color = global_color;
    }
}
