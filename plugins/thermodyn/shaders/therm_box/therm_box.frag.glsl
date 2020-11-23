#extension GL_ARB_explicit_attrib_location : enable

layout(location=0) out vec4 out_frag_color;

in vec4 color;

void main() {
    out_frag_color = color;
}