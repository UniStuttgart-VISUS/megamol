#version 130

//layout (location=0) in vec3 in_pos;

out vec3 ws_pos;

uniform mat4 modelview;
uniform mat4 project;

void main () {
// vec4[4] vertices = {
//      vec4(-1, -1, 0, 1),
//      vec4(1, -1, 0, 1),
//      vec4(-1, 1, 0, 1),
//      vec4(1, 1, 0, 1)    
// };

    //ws_pos = in_pos;
    //ws_pos = vertices[gl_VertexID].xyz;
    ws_pos = gl_Vertex.xyz;

    //gl_Position = project*modelview*vec4(ws_pos, 1.0);
    gl_Position = vec4(ws_pos, 1.0);
}