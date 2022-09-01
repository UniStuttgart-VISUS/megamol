#version 450

#include "common/common.inc.glsl"
//#include "mmstd_gl/common/tflookup.inc.glsl"
//#include "mmstd_gl/common/tfconvenience.inc.glsl"

out vec2 uvCoords;
layout (binding=7) uniform usampler2DArray imgRead;
uniform int axPxHeight;
uniform int axPxWidth;

void main() {
    mat4 compMx = projMx * viewMx;
    vec2 coord = margin.x + vec2(float(gl_VertexID % 2) * axisDistance * (dimensionCount-1), float(gl_VertexID / 2) * axisHeight);
    coord = (compMx * vec4(coord, 0, 1)).xy;
    gl_Position = vec4(coord, 0.0f, 1.0f);
    uvCoords = vec2(float(gl_VertexID % 2) * (dimensionCount-1), float(gl_VertexID / 2));
    //gl_Position =  vec4(vec2(float(gl_VertexID % 2), float((gl_VertexID % 4) / 2)),0,1);
    //uvCoords = vec2(float(gl_VertexID % 2), float((gl_VertexID % 4) / 2));
}
