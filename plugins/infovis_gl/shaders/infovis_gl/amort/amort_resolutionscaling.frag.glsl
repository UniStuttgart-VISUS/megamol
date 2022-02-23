#version 430

uniform sampler2D src_tex2D;

layout (binding=6, rgba8) uniform image2D Store;
layout (binding=7, rgba8) uniform image2D target;
layout (binding=2, rgba8) uniform image2D distancesR;
layout (binding=3, rgba8) uniform image2D distancesW;

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
    vec4 tempDist = vec4(0);
    int line = int(uv_coord.y * h) % amortLevel;
    int col = int(uv_coord.x * w) % amortLevel;
    vec4 p = vec4(2*uv_coord-vec2(1.0), 0.0, 1.0);
    ivec2 iCoord = ivec2(int(uv_coord.x * w), int(uv_coord.y * h));
    vec2 moveP = (moveM * p).xy;
    ivec2 movedICoord = ivec2((moveP.x / 2.0 + 0.5) * w, (moveP.y / 2.0 + 0.5) * h);
    int i = (amortLevel * line + col);

    float sx = float(frametype % amortLevel);
    float sy = floor(float(frametype) / float(amortLevel));
    float dist = sqrt((col-sx)*(col-sx) + (line - sy) * (line - sy)) / (amortLevel);
    dist = 1.0 - dist;
    if(dist < 0.0) dist = 0.0;
    if(dist > 1.0) dist = 1.0;
    float oldDist = imageLoad(distancesR, movedICoord).r;

    if (frametype == i){
        tempColor = vec4(texelFetch(src_tex2D, iCoord /amortLevel, 0).xyz, 1.0);
        imageStore(distancesW, iCoord, vec4(1.0,0,0.0,1));
    }else{  
        if(dist > oldDist){         
            tempColor = vec4(texelFetch(src_tex2D, iCoord /amortLevel, 0).xyz, 1.0);
            imageStore(distancesW, iCoord, vec4(dist,0.0,0.0,1.0));
        }else{
            imageStore(distancesW, iCoord, vec4(oldDist,0.0,0.0,1.0));
            tempColor = imageLoad(Store, movedICoord);
        }
    }
    //imageStore(distancesW, iCoord, vec4(dist,0,0,1));
    imageStore(target, iCoord, tempColor);
    //tempColor = imageLoad(distancesR, iCoord);
    //tempColor = vec4(tempColor.r * 50.0, 0, 0, 1.0);
    frag_out = tempColor;
}
