#version 430

//#version 440
uniform sampler2D src_tex2D;

layout (binding=5, rgba32f) uniform image2D StoreA;
layout (binding=6, rgba32f) uniform image2D StoreB;

//layout (binding = 4, rgba32f) uniform image2DArray StoreArray;

uniform int frametype;
uniform int parity;
uniform int h;
uniform int w;
uniform int oh;
uniform int ow;
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
    vec4 scaleOffsets[5];
    scaleOffsets[0] = vec4(0, 0, 0, 0);
    scaleOffsets[1] = vec4(0.125/w, 0.125/h, 0, 0);
    scaleOffsets[2] = vec4(-0.125/w, 0.125/h, 0, 0);
    scaleOffsets[3] = vec4(-0.125/w, -0.125/h, 0, 0);
    scaleOffsets[4] = vec4(0.125/w, -0.125/h, 0, 0);

    if (parity == 0){
        vec4 tempColor = vec4(0, 0, 0, 0);
        for (int q = 0; q < 5; q++){
            vec2 moveP = (moveM * (p + scaleOffsets[q])).xy;
            ivec2 movedICoord = ivec2((moveP.x / 2 + 0.5) * w, (moveP.y / 2 + 0.5) * h);
            tempColor += 0.2 * imageLoad(StoreB, movedICoord);
        }
        imageStore(StoreA, iCoord, tempColor);
        if (frametype == i){
            imageStore(StoreA, iCoord, texelFetch(src_tex2D, iCoord /amortLevel, 0));
        }
        frag_out = imageLoad(StoreA, iCoord);
    } else {
        vec4 tempColor = vec4(0, 0, 0, 0);
        for (int q = 0; q < 5; q++){
            vec2 moveP = (moveM * (p + scaleOffsets[q])).xy;
            ivec2 movedICoord = ivec2((moveP.x / 2 + 0.5) * w, (moveP.y / 2 + 0.5) * h);
            tempColor += 0.2 * imageLoad(StoreA, movedICoord);
        }
        imageStore(StoreB, iCoord, tempColor);
        if (frametype == i){
            imageStore(StoreB, iCoord, texelFetch(src_tex2D, iCoord/amortLevel, 0));
        }
        frag_out = imageLoad(StoreB, iCoord);
    }
    //imageStore(StoreArray, ivec3(iCoord.x, iCoord.y, frametype % 2), imageLoad(StoreB, iCoord));
    //frag_out = texelFetch(src_tex2D, iCoord/amortLevel, 0);
}
