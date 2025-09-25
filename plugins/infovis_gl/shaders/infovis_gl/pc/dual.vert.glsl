#version 450

#include "common/common.inc.glsl"
#include "mmstd_gl/common/tflookup.inc.glsl"
#include "mmstd_gl/common/tfconvenience.inc.glsl"

out vec2 uvCoords;
layout (binding=7, r32ui) uniform uimage2DArray imgRead;
uniform int axPxHeight;
uniform int fboHeight;
uniform int fboWidth;
uniform float debugFloat;
out vec4 color;

void main() {
    int right = (gl_InstanceID % axPxHeight);
    int cDim = gl_InstanceID / (axPxHeight*axPxHeight);
    int left = (gl_InstanceID % (axPxHeight*axPxHeight)) / axPxHeight;
    float fHeight = float(axPxHeight);
    vec2 coord = vec2(float(gl_VertexID % 2), float((gl_VertexID % 4) / 2));
    mat4 compMx = projMx * viewMx;
    vec2 realOffset = vec2(margin.x + cDim * axisDistance, margin.y);

    if (imageLoad(imgRead, ivec3(left, right, cDim)).x > 0u) {
        float ycoordinate = axisHeight * (gl_VertexID * (right/fHeight) + ((gl_VertexID + 1) % 2) * (left/fHeight)) + margin.y;
        ycoordinate = (compMx * vec4(0, ycoordinate, 0, 1)).y;
        float xcoordinate = margin.x + (gl_VertexID + cDim) * axisDistance;
        xcoordinate = (compMx * vec4(xcoordinate, 0, 0, 1)).x;
        color = vec4(imageLoad(imgRead, ivec3(left, right, cDim)).r,0,0,1);
        //color = vec4(1,0,0,1);
        gl_Position = vec4(xcoordinate, ycoordinate, 0.0f, 1.0f);
    } else {
        gl_Position = vec4(-100, -100, -1.0, 0.0);
    }

    uvCoords = coord;
}
