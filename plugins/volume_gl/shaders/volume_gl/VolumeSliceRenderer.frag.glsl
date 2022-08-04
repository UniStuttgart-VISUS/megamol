#version 430

uniform sampler2D src_tx2D;

in vec2 uv_coord;

out vec4 frag_out;

void main() {
    frag_out = texture(src_tx2D, uv_coord);
}
