#version 430

uniform mat4 MVP;

layout(lines) in;
layout(line_strip, max_vertices = 4) out;

void main() {
    for(int i = 0; i < gl_in.length(); i++) {
        gl_Position = MVP * gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();
}
