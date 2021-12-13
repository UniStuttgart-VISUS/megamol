#version 430

uniform sampler2D src_tex2D;

layout (binding=6) uniform sampler2D Store;
layout (binding=7, rgba32f) uniform image2D target;

uniform int amortLevel;
uniform int w;
uniform int h;
uniform int frametype;
uniform mat4 moveM;

in vec2 uv_coord;
//layout(early_fragment_tests) in;
out vec4 frag_out;

void main() {
    vec4 tempColor = vec4(0, 0, 0, 0);

    int line = int(uv_coord.y * h) % amortLevel;
    int col = int(uv_coord.x * w) % amortLevel;
    vec4 p = vec4(2*uv_coord-vec2(1.0), 0.0, 1.0);
    //int i = (amortLevel * (line+1) - 1 - col);
    ivec2 iCoord = ivec2(int(uv_coord.x * w), int(uv_coord.y * h));
    vec2 moveP = (moveM * p).xy;
    //ivec2 movedICoord = ivec2((moveP.x / 2.0 + 0.5) * w, (moveP.y / 2.0 + 0.5) * h);
    vec2 movedCoord = vec2(moveP.x / 2.0 + 0.5, moveP.y / 2.0 + 0.5);
    int i = (amortLevel * line + col);


    if (frametype == i){
        tempColor = texelFetch(src_tex2D, iCoord /amortLevel, 0);
    }else{
        tempColor = texture(Store, movedCoord);
    }
    imageStore(target, iCoord, tempColor);
    frag_out = tempColor;
    //frag_out = texelFetch(src_tex2D, iCoord /amortLevel, 0);
}
