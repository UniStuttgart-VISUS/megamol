out vec4 frag_color;

in vec3 color;

void main(void) {
    frag_color = vec4(color, 0.1);
}