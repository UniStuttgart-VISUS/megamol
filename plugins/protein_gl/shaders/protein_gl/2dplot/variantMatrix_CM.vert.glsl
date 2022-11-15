#version 120

attribute float val;
uniform float minVal;
uniform float maxVal;
varying float valFrag;
void main() {
    gl_Position = gl_ModelViewProjectionMatrix*gl_Vertex;
    if (gl_Position.x<0) valFrag = minVal;
    else  valFrag = maxVal;
}
