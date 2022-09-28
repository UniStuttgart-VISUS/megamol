#version 400

layout(location = 0) out vec4 frag_color;

in vec4 color;
in float radius;

void main() {
    frag_color = color;
}
