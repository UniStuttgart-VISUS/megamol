attribute vec3 pos;
attribute float flag;

varying vec3 lightDir;
varying vec3 view;
varying vec3 posWS;
varying float flagFrag;

void main() {

    // Vertex positions
    gl_Position = gl_ModelViewProjectionMatrix*vec4(pos, 1.0);
    posWS = pos;

    // Get view vector in eye space
    view = (gl_ModelViewMatrix*vec4(pos, 1.0)).xyz;

    // Transformation of normal into eye space
    //normalFrag = gl_NormalMatrix*normal;

    // Get the direction of the light
    //lightDir = vec3(gl_LightSource[0].position)-view;
    lightDir = gl_LightSource[0].position.xyz;

    // Vertex flag
    flagFrag = flag;
}
