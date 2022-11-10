#version 120

attribute vec3 pos;
attribute vec3 normal;
attribute vec3 texCoord;

varying vec3 lightDir;
varying vec3 view;
varying vec3 normalFrag;
varying vec3 posWS;

void main() {

    // Vertex positions
    gl_Position = gl_ModelViewProjectionMatrix*vec4(pos, 1.0);
    //gl_Position = gl_ModelViewProjectionMatrix*gl_Vertex; // Vertex positions
    posWS = pos;

    // Get view vector in eye space
    view = (gl_ModelViewMatrix*vec4(pos, 1.0)).xyz;

    // Transformation of normal into eye space
    normalFrag = gl_NormalMatrix*normal;

    // Get the direction of the light
    //lightDir = vec3(gl_LightSource[0].position)-view;
    lightDir = gl_LightSource[0].position.xyz;

    // Texture coordinate
    gl_TexCoord[0].stp = texCoord;
}
