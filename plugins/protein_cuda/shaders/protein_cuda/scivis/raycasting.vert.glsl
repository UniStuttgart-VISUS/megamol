#version 130

varying vec3 posES;
varying vec4 pos;
void main(void) {
    gl_Position = gl_ModelViewProjectionMatrix*gl_Vertex;
    pos = gl_Position;
    posES = (gl_ModelViewMatrix*gl_Vertex).xyz;
    gl_TexCoord[0] = gl_MultiTexCoord0;
}
