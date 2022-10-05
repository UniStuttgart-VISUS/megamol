#version 120

#include "protein_cuda/electrostatics/colorMix.glsl"
#include "protein_cuda/electrostatics/ppLighting.glsl"
#include "protein_gl/colormaps/RGB2MSH.glsl"
#include "protein_gl/colormaps/MSH2RGB.glsl"
#include "protein_gl/colormaps/COOLWARM.glsl"
#include "protein_gl/colormaps/HSV2RGB.glsl"

uniform sampler3D potentialTex;
uniform int colorMode;
uniform int renderMode;
uniform vec3 colorMin;
uniform vec3 colorMax;
uniform vec3 colorUniform;
uniform float minPotential;
uniform float maxPotential;
uniform float alphaScl;

varying vec3 lightDir;
varying vec3 view;
varying vec3 normalFrag;
varying vec3 posWS;

void main() {

    // DEBUG clipping planes
    //if (posWS.z < 49) return;
    //if (posWS.z > 50) return;

    //if (posWS.z > 50) return;

    vec4 lightparams, color;

    // Determine lighting parameters
    if (renderMode == 1) { // Points
        lightparams = vec4(1.0, 0.0, 0.0, 1.0);
    } else if (renderMode == 2) { // Wireframe
        lightparams = vec4(1.0, 0.0, 0.0, 1.0);
    } else if (renderMode == 3) { // Surface
        lightparams = vec4(0.2, 0.8, 0.0, 10.0);
    }

    // Determine color
    if (colorMode == 0) { // Uniform color
        color = vec4(colorUniform, 1.0);
        //color = vec4(0.0, 0.0, 0.0, 1.0);
    } else if (colorMode == 1) { // Normal
        lightparams = vec4(1.0, 0.0, 0.0, 1.0);
        color = vec4(normalize(normalFrag), 1.0);
    } else if (colorMode == 2) { // Texture coordinates
        lightparams = vec4(1.0, 0.0, 0.0, 1.0);
        color = vec4(gl_TexCoord[0].stp, 1.0);
    } else if (colorMode == 3) { // Surface potential

        // Interpolation in MSH color space
        vec3 colMsh = CoolWarmMsh(texture3D(potentialTex, gl_TexCoord[0].stp).a,
                        minPotential, 0.0, maxPotential);
        //vec3 colMsh = vec3(1.0, 1.0, 0.0);
        color = vec4(MSH2RGB(colMsh.x, colMsh.y, colMsh.z), 1.0);

    } else { // Invalid color mode
        color = vec4(0.5, 1.0, 1.0, 1.0);
    }

    // Apply scaling of alpha value
    color.a *= alphaScl;

    gl_FragColor = vec4(LocalLighting(normalize(view), normalize(normalFrag),
            normalize(lightDir), color.rgb, lightparams), color.a);
}
