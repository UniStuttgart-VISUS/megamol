#version 450
#include "common/common.inc.glsl"
#include "core/tflookup.inc.glsl"
#include "core/tfconvenience.inc.glsl"
in vec2 uvCoords;
out vec4 fragOut;
uniform int axPxHeight;
uniform int axPxWidth;
uniform float debugFloat;
layout (binding=7) uniform usampler2DArray imgRead;

void main(){
    int axPxDist = axPxWidth / int(dimensionCount-1);

    int cdim = int(floor(uvCoords.x));
    float relx = fract(uvCoords.x);
    float result = 0;

    for (int i = 0; i < axPxHeight; i++){
        //int((theta + 45/360 * 2 * 3.141) / (90/360 * 2 * 3.141) * axPxHeight
        float theta = (float(i)/float(axPxHeight) * 3.141/2.0 + 3.141/4.0);
        vec2 orth = vec2(cos(theta), sin(theta));
        vec2 dir = vec2(-orth.y, orth.x);
        float rho = (dir.x * uvCoords.y - dir.y * relx) / (dir.x * orth.y - dir.y * orth.x);
        if(texelFetch(imgRead, ivec3(i, rho * axPxHeight, cdim), 0).x > 0){
            result += 3 * texelFetch(imgRead, ivec3(i, rho * axPxHeight, cdim), 0).x;
        }
    }
    if(result > 0){
            fragOut = tflookup(result);
        }else{
            discard;
        }
}
