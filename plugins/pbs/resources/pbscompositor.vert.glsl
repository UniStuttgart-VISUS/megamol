#version 430

layout (location=0) in vec3 in_pos;

out vec3 ws_pos;

// const vec4[4] vertices = {
//     vec4(-1, -1, 0, 1),
//     vec4(1, -1, 0, 1),
//     vec4(-1, 1, 0, 1),
//     vec4(1, 1, 0, 1)    
// };

uniform mat4 modelview;
uniform mat4 project;

void main () {
    ws_pos = in_pos;

    //gl_Position = project*modelview*vec4(ws_pos, 1.0);
    gl_Position = vec4(ws_pos, 1.0);
}