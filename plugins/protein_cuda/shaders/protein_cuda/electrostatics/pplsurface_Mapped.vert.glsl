#version 120

uniform sampler3D potentialTex0;
uniform sampler3D potentialTex1;

attribute vec3 posNew;
attribute vec3 normal;
attribute vec3 texCoordNew;
attribute float corruptTriangleFlag;
attribute float pathLen;
attribute float surfAttrib;

varying vec3 lightDir;
varying vec3 view;
varying vec3 normalFrag;
varying vec3 posNewFrag;
varying float pathLenFrag;
varying float surfAttribFrag;
varying float corruptFrag;

void main() {

    // Vertex positions
    gl_Position = gl_ModelViewProjectionMatrix*vec4(posNew, 1.0);

    // Get view vector in eye space
    view = (gl_ModelViewMatrix*vec4(posNew, 1.0)).xyz;

    // Transformation of normal into eye space
    normalFrag = gl_NormalMatrix*normal;

    // Get the direction of the light
    // Note: is already transformed using the modelview matrix when calling glLight
    lightDir = gl_LightSource[0].position.xyz;

    // Texture coordinates
    gl_TexCoord[0].stp = texCoordNew;

    // Object space positions
    posNewFrag = posNew;

    // Path length
    pathLenFrag = pathLen;

    // Surface attribute
    surfAttribFrag = surfAttrib;

    corruptFrag =corruptTriangleFlag;
}
