#version 460

layout(location = 0) out vec4 color_out;

uniform sampler2D bwtex;
uniform vec4 input_color;

in vec2 texcoord;

void main(void) {
    vec4 col = texture(bwtex,texcoord).rgba;
    if(col.x < 0.00001){
        discard;
    }
    color_out = vec4(input_color.xyz, input_color.w * col.x);
}
