//#version 440
layout (binding = 4) uniform sampler2D src_tex2D;

layout (binding=5, rgba32f) uniform image2D StoreA;
layout (binding=6, rgba32f) uniform image2D StoreB;

uniform int frametype;
uniform int h;
uniform int w;
uniform int approach;
uniform int amortLevel;
uniform mat4 moveM;

in vec2 uv_coord;
//layout(early_fragment_tests) in;
out vec4 frag_out;

void main() {
    int line = int(uv_coord.y * h) % amortLevel;
    int col = int(uv_coord.x * w) % amortLevel;
    vec4 p = vec4(2*uv_coord-vec2(1.0), 0.0, 1.0);
    //int i = (amortLevel * (line+1) - 1 - col);
    ivec2 iCoord = ivec2(int(uv_coord.x * w), int(uv_coord.y * h));
    vec2 moveP = (moveM * p).xy;
    ivec2 movedICoord = ivec2((moveP.x / 2 + 0.5) * w, (moveP.y / 2 + 0.5) * h);
    int i = (amortLevel * line + col);

    if (frametype % 2 == 0){
        imageStore(StoreA, iCoord, imageLoad(StoreB, movedICoord));
        if (frametype == i){
            imageStore(StoreA, iCoord, texture(src_tex2D, uv_coord));
        }
        frag_out = imageLoad(StoreA, iCoord);
    } else {
        imageStore(StoreB, iCoord, imageLoad(StoreA, movedICoord));
        if (frametype == i){
            imageStore(StoreB, iCoord, texture(src_tex2D, uv_coord));
        }
        frag_out = imageLoad(StoreB, iCoord);
    }
    //imageStore(StoreArray, ivec3(iCoord.x, iCoord.y, frametype % 2), imageLoad(StoreB, iCoord));
    //frag_out = imageLoad(StoreArray, ivec3(iCoord.x, iCoord.y, frametype % 2));
    //frag_out = texture(src_tex2D, uv_coord);
}
