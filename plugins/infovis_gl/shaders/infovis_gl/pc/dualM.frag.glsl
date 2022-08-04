#version 450
#include "common/common.inc.glsl"
#include "mmstd_gl/common/tflookup.inc.glsl"
#include "mmstd_gl/common/tfconvenience.inc.glsl"
in vec2 uvCoords;
out vec4 fragOut;
uniform int axesHeight;
layout (binding=7, r32ui) uniform uimage2DArray imgRead;

void main(){
    int cdim = int(uvCoords.x * (dimensionCount-1));
    float relX = fract(uvCoords.x);
    float relY = uvCoords.y;
    float result = 0;
    for (int i = 0; i < axesHeight; i++){
        int targetY = int((i - relY) / relX);
        result += imageLoad(imgRead, ivec3(i, axesHeight - targetY, cdim)).x;
    }
    if(result > 0){
        //fragOut = tflookup(result);
    }else{
        //discard;
    }
    //fragOut = vec4(relX, relX,relX,1);
}
