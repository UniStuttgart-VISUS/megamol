#version 450

#include "common/common.inc.glsl"
#include "mmstd_gl/common/tflookup.inc.glsl"
#include "mmstd_gl/common/tfconvenience.inc.glsl"

in vec2 uvCoords;
out vec4 fragOut;
uniform int axPxHeight;
uniform int axPxWidth;
uniform float debugFloat;
layout (binding=7) uniform usampler2DArray imgRead;

void main() {
    int axPxDist = axPxWidth / int(dimensionCount-1);

    int cdim = int(floor(uvCoords.x) / float(axPxDist));
    int a, b, c, d;

    float result = 0;
    //if (float(int(uvCoords.x) % axPxDist) / float(axPxDist) > 0.5) {
    for (int i = 0; i < axPxHeight; i++) {
        float xbot = (int(floor(uvCoords.x)) % axPxDist) / float(axPxDist);
        float xtop = (int(ceil(uvCoords.x)) % axPxDist) / float(axPxDist);
        a = int((floor(uvCoords.y) - float(i)) / xbot); // bottom left
        b = int((floor(uvCoords.y) - float(i)) / xtop); // bottom right
        c = int((ceil(uvCoords.y) - float(i)) / xbot); // bottom left
        d = int((ceil(uvCoords.y) - float(i)) / xtop); // bottom right
        int minY = a;
        int maxY = a;
        maxY = max(max(a, b), max(c, d));
        minY = min(min(a, b), min(c, d));
        if (minY < -i) {
            minY = -i;
        }
        if (maxY > axPxHeight - i) {
            maxY = axPxHeight - i;
        }
        while (minY <= maxY && minY < axPxHeight) {
            minY = minY + 1;
            result += 10 * texelFetch(imgRead, ivec3(i, i + minY, cdim), 0).x;
        }
    }
    //}else{
    //}

    if (result > 0) {
        fragOut = tflookup(result);
    } else {
        fragOut = vec4(0.0);
    }
    //fragOut = vec4(fract(uvCoords.x), uvCoords.y / float(axPxHeight) , 0 ,1);
    if (uvCoords.y == 1.0) {
        //vec4(1.0);
    } else {
        //vec4(0.0, 0.0, 0.0, 1.0);
    }

    //fragOut = vec4(relX, relY,0,1);

}
