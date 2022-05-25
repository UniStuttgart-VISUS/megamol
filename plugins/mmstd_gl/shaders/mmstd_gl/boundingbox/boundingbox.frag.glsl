#version 330

layout(location = 0) out vec4 frag_color;

uniform vec3 color = vec3(1.0);

void main(void) {
    frag_color = vec4(color, 1.0);
}
