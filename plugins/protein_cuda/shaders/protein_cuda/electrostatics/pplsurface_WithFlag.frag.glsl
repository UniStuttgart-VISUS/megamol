#version 120

#include "protein_cuda/electrostatics/colorMix.glsl"
#include "protein_cuda/electrostatics/ppLighting.glsl"
#include "protein_gl/colormaps/RGB2MSH.glsl"
#include "protein_gl/colormaps/MSH2RGB.glsl"
#include "protein_gl/colormaps/COOLWARM.glsl"
#include "protein_gl/colormaps/HSV2RGB.glsl"

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
