#version 450

uniform mat4 view_mx;
uniform mat4 proj_mx;

layout(location = 0) in vec3 v_position;
layout(location = 1) in vec4 v_color;

out vec4 color;

void main()
{
    //gl_Position = vec4(v_position.xy, 0.5 ,1.0);
    gl_Position =  proj_mx * view_mx * vec4(v_position, 1.0);
    color = v_color;
}
