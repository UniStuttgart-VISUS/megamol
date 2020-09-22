uniform sampler2D samplers[4];
uniform mat4 mMatrices[4];

uniform int frametype;
uniform int h;
uniform int w;
uniform int approach;

in vec2 uv_coord;
//layout(early_fragment_tests) in;
out vec4 frag_out;

void main()
{
    int line = int(uv_coord.y * h);
    int col = int(uv_coord.x * w);
    int index = line * 2 + col;
    vec4 p = vec4(2*uv_coord-vec2(1.0), 0.0, 1.0);
    if(line % 2 == 1){
        if(col % 2 == 0){
            frag_out = texture(samplers[3], 0.5 * (mMatrices[3] * p).xy + vec2(0.5));
        }else{
            frag_out = texture(samplers[2], 0.5 * (mMatrices[2] * p).xy + vec2(0.5));
        }
    } else {
        if(col % 2 == 1){
            frag_out = texture(samplers[0], 0.5 * (mMatrices[0] * p).xy + vec2(0.5));
        }else{
            frag_out = texture(samplers[1], 0.5 * (mMatrices[1] * p).xy + vec2(0.5));
        }  
    }
}