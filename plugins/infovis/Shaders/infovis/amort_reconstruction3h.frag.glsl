#version 430

uniform sampler2DArray tx2D_array;
uniform sampler2D history_tx2D;
uniform sampler2D result_tx2D;

uniform int frametype;
uniform int h;
uniform int w;
uniform int approach;

in vec2 uv_coord;
//layout(early_fragment_tests) in;
out vec4 frag_out;

void main() {
    frag_out = texture(result_tx2D, uv_coord);
}
