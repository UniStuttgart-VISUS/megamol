#version 430

layout (depth_greater) out float gl_FragDepth; // we think this is right
// this should be wrong //layout (depth_less) out float gl_FragDepth;
#extension GL_ARB_explicit_attrib_location : enable

uniform vec4 lineColor = vec4(1.0, 0.75, 0.2, 1.0);

out layout(location = 0) vec4 outCol;

void main(void) {
    outCol = lineColor;
}