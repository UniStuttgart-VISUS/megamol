#version 430

uniform mat4 MVP;
uniform mat4 MVinvtrans;

layout(triangles) in;
layout(triangle_strip, max_vertices = 4) out;

in vec4 color[];
in vec3 n[];
out vec4 mycol;
out vec3 rawnormal;

void main() {
    for(int i = 0; i < gl_in.length(); i++) {
        gl_Position = MVP * gl_in[i].gl_Position;
        mycol = color[i];
        vec4 normal4 = MVinvtrans * vec4(n[i], 0);
        rawnormal = normalize(n[i]);
        EmitVertex();
    }
    EndPrimitive();
}
