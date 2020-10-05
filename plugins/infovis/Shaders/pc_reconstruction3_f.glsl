uniform sampler2DArray tx2D_array;

uniform int frametype;
uniform int h;
uniform int w;
uniform int approach;

uniform mat4 mMa;
uniform mat4 mMb;
uniform mat4 mMc;
uniform mat4 mMd;

uniform mat4 mMatrices[4];

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
            p = mMd * p;
        }else{
            t = 2;
            p = mMc * p;
        }
    } else {
        if(col % 2 == 1){
            t = 0;
            p = mMa * p;
        }else{
            t = 1;
            p = mMb * p;
        }  
    }
    for (int i = 0; i < 25; i++){
        frag_out += texture(tx2D_array, vec3(0.5 *  p.xy + vec2(0.5), t + (i*4))) / 25;
    }
}