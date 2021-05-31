#version 430

uniform sampler2DArray src_tex2D;

uniform int frametype;
uniform int h;
uniform int w;
uniform int approach;

uniform mat4 moveMatrices[4];

in vec2 uv_coord;
//layout(early_fragment_tests) in;
out vec4 frag_out;

void main() {
    int line = int(uv_coord.y * h) % 2;
    int col = int(uv_coord.x * w) % 2;
    vec4 p = vec4(2*uv_coord-vec2(1.0), 0.0, 1.0);
    int i = (2 * line + 1 - col);
    frag_out = texture(src_tex2D, vec3(0.5 * (moveMatrices[i] * p).xy + vec2(0.5), i));
}
