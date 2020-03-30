uniform sampler2D texSampler;

in vec2 texCoord;

layout (location = 0) out vec4 out_frag_color; 

void main(void) {
    vec4 col = texture(texSampler, texCoord);
    out_frag_color = col;
}