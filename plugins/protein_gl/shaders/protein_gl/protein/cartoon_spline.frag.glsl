/*
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS) / Michael Krone. Alle Rechte vorbehalten.
 */

//#version 120

varying vec3 normal;
varying vec3 color;

void main(void) {
#if 0
    vec3 norm, lightDir;
    vec4 diffuse, ambient, globalAmbient;
    float NdotL;

    norm = normalize(gl_NormalMatrix * normal);
    lightDir = normalize(vec3(gl_LightSource[0].position));
    NdotL = max(dot(norm, lightDir), 0.0);
    diffuse = vec4( color, 1.0) * gl_LightSource[0].diffuse;

    /* Compute the ambient and globalAmbient terms */
    ambient = vec4( color, 1.0) * gl_LightSource[0].ambient;
    globalAmbient = gl_LightModel.ambient * vec4( color, 1.0);
    gl_FragColor = NdotL * diffuse + globalAmbient + ambient;
#else
    // normalize the direction of the light
    vec3 lightDir = normalize(vec3(gl_LightSource[0].position));

    // normalize the halfVector to pass it to the fragment shader
    vec3 halfVector = normalize(gl_LightSource[0].halfVector.xyz);

    vec4 diffuse, ambient;
    // compute the diffuse, ambient and globalAmbient terms
    diffuse = vec4( color, 1.0) * gl_LightSource[0].diffuse;
    ambient = vec4( color, 1.0) * gl_LightSource[0].ambient;
    ambient += gl_LightModel.ambient * vec4( color, 1.0);

    vec3 n;
    float NdotL,NdotHV;

    // store the ambient term
    vec4 colorOut = ambient;

    // transformation of the normal into eye space
    n = normalize(gl_NormalMatrix * normal);

    // compute the dot product between normal and lightDir
    NdotL = dot(n,lightDir);
    if (NdotL > 0.0) {
        // front side
        colorOut += diffuse * NdotL;
        NdotHV = max(dot(n,halfVector),4.88e-04);
        colorOut += gl_FrontMaterial.specular * gl_LightSource[0].specular * pow(NdotHV, gl_FrontMaterial.shininess);
    }

    gl_FragColor = colorOut;
#endif
    //gl_FragColor = gl_Color;
}
