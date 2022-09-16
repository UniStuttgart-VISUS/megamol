varying vec3 lightDir;
varying vec3 view;
varying vec3 posWS;
varying float flagFrag;

void main() {

    vec4 lightparams;
    vec3 color;

    lightparams = vec4(1.0, 0.0, 0.0, 1.0);

    color = flagFrag*(vec3(1.0, 0.0, 0.0)) + (1.0-flagFrag)*vec3(0.7, 0.8, 1.0);
    //color = vec3(1.0, 0.0, 0.0);

    //gl_FragColor = vec4(LocalLighting(normalize(view), normalize(normalFrag),
    //        normalize(lightDir), color.rgb, lightparams), 1.0);


    gl_FragColor = vec4(color.rgb, 1.0);
}
