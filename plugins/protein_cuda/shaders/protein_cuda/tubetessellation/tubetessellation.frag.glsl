#version 140

#include "common_defines_btf.glsl"
#include "lightdirectional.glsl"

#extension GL_ARB_conservative_depth:require
layout (depth_greater) out float gl_FragDepth; // we think this is right
// this should be wrong //layout (depth_less) out float gl_FragDepth;
#extension GL_ARB_explicit_attrib_location : enable

uniform mat4 MVP;
uniform mat4 MVPinv;
uniform mat4 MVPtransp;

uniform vec4 viewAttr;

FLACH in vec4 objPos;
FLACH in vec4 camPos;
FLACH in vec4 lightPos;
FLACH in float squarRad;
FLACH in float rad;
FLACH in vec4 vertColor;

in vec4 myColor;

out layout(location = 0) vec4 outCol;

void main(void) {
    outCol = myColor;
}
