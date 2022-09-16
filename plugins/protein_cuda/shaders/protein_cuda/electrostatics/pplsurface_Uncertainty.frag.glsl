varying vec3 lightDir;
varying vec3 view;
varying vec3 posWS;
varying float uncertaintyFrag;
varying vec3 normalFrag;

uniform float maxUncertainty;

void main() {

    vec4 lightparams;
    vec4 color;

    //lightparams = vec4(0.2, 0.8, 0.0, 10.0);
    //lightparams = vec4(0.2, 0.0, 0.0, 10.0);
    vec3 colOrangeMsh = vec3(100, 0.9746*uncertaintyFrag/maxUncertainty, 0.8968);
        //vec3 colYellowMsh = vec3(100, 0.81*potDiff/(maxPotential-minPotential), 1.7951);
        //vec3 blueMsh = vec3(90, 1.08*potDiff/(maxPotential-minPotential), -1.1);

    color = vec4(MSH2RGB(colOrangeMsh.r, colOrangeMsh.g, colOrangeMsh.b), 1.0);

    //vec3 colMsh = CoolWarmMsh(uncertaintyFrag, 0.0, maxUncertainty/2.0, maxUncertainty);
    //color = MSH2RGB(colMsh.x, colMsh.y, colMsh.z);

    //gl_FragColor = vec4(LocalLighting(normalize(view), normalize(normalFrag),
    //        normalize(lightDir), color.rgb, lightparams), 1.0);

    gl_FragColor = color;

}
