uniform sampler2D input_tx2D;

in vec2 uv_coord;

out layout(location = 0) vec4 frag_out;


void main(void) {
    frag_out = texture(input_tx2D,uv_coord).rgba;
}