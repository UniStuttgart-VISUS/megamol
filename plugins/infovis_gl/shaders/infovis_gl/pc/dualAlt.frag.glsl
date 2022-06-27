#version 450
#include "common/common.inc.glsl"
#include "core/tflookup.inc.glsl"
#include "core/tfconvenience.inc.glsl"
in vec2 uvCoords;
out vec4 fragOut;
uniform int axesHeight;
layout (binding=7, r32ui) uniform uimage2DArray imgRead;

void main(){
    int cdim = int(uvCoords.x * (dimensionCount-1));
    float relX = uvCoords.x * (dimensionCount-1) - cdim;
    float relY = uvCoords.y;
    int result = 0;
    for (int i = 0; i < axesHeight; i++){
        float checkheight = float(i) / float(axesHeight);
        float deltaY = checkheight - relY;
        float deltaX = 0.0f - relX;
        float targetY = checkheight + deltaY / deltaX;
        int targetPixel = int(targetY * axesHeight);
        if(targetPixel < axesHeight && targetPixel > 0){
            result += int(imageLoad(imgRead, ivec3(i, targetY * axesHeight, cdim)).x);
            
        }
    }
    if(result > 0){
        fragOut = tflookup(float(result));
    }else{
        discard;
    }
    //fragOut = result * vec4(1.0/5000.0, 1.0/5000.0, 1.0/5000.0, 1);
    //fragOut = vec4(relX, relY,0,1);
    
}
