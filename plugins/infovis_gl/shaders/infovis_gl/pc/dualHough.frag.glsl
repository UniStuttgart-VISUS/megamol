#version 450

#include "common/common.inc.glsl"
#include "mmstd_gl/common/tflookup.inc.glsl"
#include "mmstd_gl/common/tfconvenience.inc.glsl"

in vec2 uvCoords;
out vec4 fragOut;
// height is actually number of bins
uniform int axPxHeight;
uniform int axPxWidth;
uniform float debugFloat;
uniform int binsNr;
uniform bool selectMode;
uniform int thetas;
uniform int rhos;
layout (binding=7) uniform usampler2DArray imgRead;

void main() {
    int axPxDist = axPxWidth / int(dimensionCount-1);
    float PI = 3.1415926535;
    int cdim = int(floor(uvCoords.x));
    float relx = fract(uvCoords.x);
    int result = 0;

    for (int i = 0; i <= thetas; i++) {
        //int((theta + 45/360 * 2 * 3.141) / (90/360 * 2 * 3.141) * axPxHeight
        float theta = (float(i)/float(thetas) * PI/2.0 + PI/4.0);
        vec2 orth = vec2(cos(theta), sin(theta));
        vec2 dir = vec2(-orth.y, orth.x);
        //float rho = (dir.x * uvCoords.y - dir.y * relx) / (dir.x * orth.y - dir.y * orth.x);
        float rho = relx * cos(theta) + fract(uvCoords.y) * sin(theta);
        if (texelFetch(imgRead, ivec3(i, rho * (rhos-1), cdim), 0).x > 0) {
            result += int(texelFetch(imgRead, ivec3(i, rho * (rhos-1), cdim), 0).x);
            //result += 1;
            //fragOut = vec4(theta/PI*2.0);
        }
    }
    /*
    if (result == 1) {
        //fragOut = tflookup(result-1);
        //fragOut = vec4(theta/(PI/2.0));
        fragOut = vec4(1,0,0,1);
    }
    if (result == 2) {
        fragOut = vec4(0,1,0,1);
    }
    if (result == 3) {
        fragOut = vec4(1,1,1,1);
    }
    if(result == 0){
     discard;
    }
    */
    if(result > 0){
        if(!selectMode){
            fragOut = tflookup(result-1);
        }else{
            fragOut = vec4(1,0,0,1);
        }
    } else {
        discard;
    }
}
