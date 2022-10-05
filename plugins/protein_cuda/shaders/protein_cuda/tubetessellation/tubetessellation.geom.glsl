#version 400

uniform mat4 MVP;

layout(triangles) in;
layout(triangle_strip, max_vertices = 4) out;

in vec4 vertColor[];
out vec4 myColor;

void main() {
  for(int i = 0; i < gl_in.length(); i++) {
    vec4 h = gl_in[i].gl_Position;
    h.z = 0.0;
    gl_Position = MVP * h;
    myColor = vertColor[i];
    EmitVertex();
  }
  EndPrimitive();
}
