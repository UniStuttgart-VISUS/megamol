//#version 120

varying vec4 diffuse,ambient;
varying vec3 normal,lightDir,halfVector;
varying vec3 view;


void main(void) {

    vec4 lightparams = vec4(0.6, 0.8, 0.4, 10.0);
    gl_FragColor = vec4(LocalLightingStreamlines(
        -view, normal,
        lightDir, gl_Color, lightparams), 1.0);
}
