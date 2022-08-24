#version 460

uniform sampler2D input_tx2D;
uniform isampler2D input_txid;

in vec2 texcoord;

out layout(location = 0) vec4 frag_out;

void main(void) {
    int id = texture(input_txid, texcoord).r;
    if(id == 0) {
        discard;
    } else {
        frag_out = texture(input_tx2D, texcoord).rgba;
    }
}
