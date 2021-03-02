//#version 440
uniform sampler2D src_tex2D;

layout (binding=0, rgba32f) uniform image2D StoreA;
layout (binding=1, rgba32f) uniform image2D StoreB;

uniform int frametype;
uniform int h;
uniform int w;
uniform int approach;
uniform int amortLevel;
uniform mat4 moveM;

in vec2 uv_coord;
//layout(early_fragment_tests) in;
out vec4 frag_out;

void main()
{
    int line = int(uv_coord.y * h) % amortLevel;
    int col = int(uv_coord.x * w) % amortLevel;
    vec4 p = vec4(2*uv_coord-vec2(1.0), 0.0, 1.0);
    //int i = (amortLevel * (line+1) - 1 - col);
    ivec2 iCoord = ivec2(int(uv_coord.x * w), int(uv_coord.y * h));
    int i = (amortLevel * line + col);
    //movewrite
    imageStore(StoreA, iCoord, imageLoad(StoreB, ivec2(w, h) * ivec2(0.5 * (moveM * p).xy + vec2(0.5))) );
    if (i == frametype){
        imageStore(StoreA, iCoord, texture(src_tex2D, p.xy));
    }else{
        frag_out = vec4(0,0,0.40,1);
    }//frag_out = vec4(0.2, 0.2, 0.2, 1);
    frag_out = imageLoad(StoreA, iCoord);
    imageStore(StoreB, iCoord, imageLoad(StoreA, iCoord));
}
