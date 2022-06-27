#version 450
#include "common/common.inc.glsl"
#include "core/tflookup.inc.glsl"
#include "core/tfconvenience.inc.glsl"
out vec2 uvCoords;
layout (binding=7, r32ui) uniform uimage2DArray imgRead;
uniform int axesHeight;
out vec4 color;

void main() {
    int right = (gl_InstanceID % axesHeight);
    int cDim = gl_InstanceID / (axesHeight*axesHeight);
    int left = (gl_InstanceID-cDim * (axesHeight*axesHeight)) / axesHeight;
    float fHeight = float(axesHeight);
    vec2 coord = vec2(float(gl_VertexID % 2), float((gl_VertexID % 4) / 2));
    mat4 compMx = projMx * viewMx;
    vec2 realOffset = vec2(margin.x + cDim * axisDistance, margin.y);

    //gl_Position = vec4(2.0f * coord - 1.0f, 0.0f, 1.0f);
    if(imageLoad(imgRead, ivec3(left, right, cDim)).x > 0u){
        //gl_Position = vec4(2.0f * coord - 1.0f, 0.0f, 1.0f);
        float ycoordinate = axisHeight * (gl_VertexID * (right/fHeight) + ((gl_VertexID + 1) % 2) * (left/fHeight)) + margin.y;
        ycoordinate = (compMx * vec4(0, ycoordinate, 0, 1)).y;
        float xcoordinate = margin.x + (gl_VertexID + cDim) * axisDistance;
        xcoordinate = (compMx * vec4(xcoordinate, 0, 0, 1)).x;
        color = tflookup(imageLoad(imgRead, ivec3(left, right, cDim)).x);
        color.a = float(imageLoad(imgRead, ivec3(left, right, cDim)).x) / 5000.0;
        gl_Position = vec4(xcoordinate, ycoordinate, 0.0f, 1.0f);
    }else{
        gl_Position = vec4(-100, -100, -1.0, 0.0);
    }
    
    uvCoords = coord;
}
