#version 450
#include "common/common.inc.glsl"
#include "core/tflookup.inc.glsl"
#include "core/tfconvenience.inc.glsl"
in vec2 uvCoords;
out vec4 fragOut;
uniform int axesHeight;
layout (binding=7, r32ui) uniform uimage2DArray imgRead;

void main(){
    int cdim = int(trunc(uvCoords.x));
    float relX = fract(uvCoords.x);
    float relY = uvCoords.y;
    float result = 0;
    for (int i = 0; i < axesHeight; i++){
        int targetY = int((floor(relY)-float(i)) / relX); // = steigung
        result += imageLoad(imgRead, ivec3(i, i + targetY, cdim)).x;
    }
    if(result > 0){
        fragOut = tflookup(result);
    }else{
        fragOut = vec4(0.0);
    }
    //fragOut = result * vec4(1.0/5000.0, 1.0/5000.0, 1.0/5000.0, 1);
    //fragOut = vec4(relX, relY,0,1);
    
}
