uniform sampler2DArray tx2D_array;

uniform int frametype;
uniform int h;
uniform int w;
uniform int approach;

uniform mat4 mMatrices[100];

in vec2 uv_coord;
//layout(early_fragment_tests) in;
out vec4 frag_out;

void main()
{
    int line = int(uv_coord.y * h);
    int col = int(uv_coord.x * w);
    int t = 0;
    vec4 p = vec4(2*uv_coord-vec2(1.0), 0.0, 1.0);
    if(line % 2 == 1){
        if(col % 2 == 0){
            t = 3;
        }else{
            t = 2;
        }
    } else {
        if(col % 2 == 1){
            t = 0;
        }else{
            t = 1;
        }  
    }
    vec4 r;
    for (int i = 0; i < 25; i++){
        r = mMatrices[t + (i * 4)] * p;
        frag_out += texture(tx2D_array, vec3(0.5 *  r.xy + vec2(0.5), t + (i*4))) / 25;
    }
}