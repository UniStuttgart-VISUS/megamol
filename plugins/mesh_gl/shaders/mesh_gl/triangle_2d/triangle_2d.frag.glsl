#version 330

#include "../triangle_common/defines.inc.glsl"

in vec4 vertex_color;

out vec4 fragColor;

void main() {
    fragColor = vertex_color;
}
