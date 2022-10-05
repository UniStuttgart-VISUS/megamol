#version 140

#include "protein_cuda/protein_cuda/commondefines.glsl"
#include "lightdirectional.glsl"

uniform mat4 modelview;
uniform mat4 proj;

in vec4 pos0;
in vec4 pos1;
in vec4 color;

out vec4 inPos0;
out vec4 inPos1;
out vec4 inColor;

void main(void) {
    inPos0 = pos0;
    inPos1 = pos1;
    inColor = color;
}
