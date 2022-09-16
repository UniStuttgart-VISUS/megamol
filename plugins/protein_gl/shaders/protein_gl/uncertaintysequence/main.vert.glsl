#version 430

#extension GL_ARB_explicit_attrib_location : enable

layout (location = 0) in vec4 vertexPosition;              // the input vertex position
out vec4 quadPos;                                          // transformed coordinates from world space to "tile" space in [0,1] for color interpolation
uniform vec2 worldPos;                                     // position of lower left corner of uncertainty visualisation tile for current aminoacid
uniform mat4 MVP;                                          // the 'self calculated' (M)odel(V)iew(P)rojection matrix


void main() {

    quadPos = vec4((vertexPosition.x - worldPos.x), (vertexPosition.y + worldPos.y), vertexPosition.z, vertexPosition.w);

    gl_Position = MVP * vertexPosition;
}
