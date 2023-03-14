#version 450

uniform usampler2D input_tx2D;

uniform sampler1D tf_tx1D;

in vec2 uv_coord;

out layout(location = 0) vec4 frag_out;

void main(void) {
    float val = float(texture(input_tx2D, uv_coord).r) / 255.0f;
    vec4 color = texture(tf_tx1D, val);
    //frag_out = vec4(val, val, val, 1.0f);
    frag_out = color;
}
