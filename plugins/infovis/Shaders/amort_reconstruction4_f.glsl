uniform sampler2DArray src_tex2D;

uniform int frametype;
uniform int h;
uniform int w;
uniform int approach;
uniform int amortLevel;
uniform mat4 moveMatrices[49];

in vec2 uv_coord;
//layout(early_fragment_tests) in;
out vec4 frag_out;

void main() {
    int line = int(uv_coord.y * h) % amortLevel;
    int col = int(uv_coord.x * w) % amortLevel;
    vec4 p = vec4(2*uv_coord-vec2(1.0), 0.0, 1.0);
    //int i = (amortLevel * (line+1) - 1 - col);
    int i = (amortLevel * line + col);
    frag_out = texture(src_tex2D, vec3(0.5 * (moveMatrices[i] * p).xy + vec2(0.5), i));
    //frag_out = vec4(0.2, 0.2, 0.2, 1);
}
