/*
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS) / Michael Krone. Alle Rechte vorbehalten.
 */

//#version 120

varying vec4 diffuse,ambient;
varying vec3 normal,lightDir,halfVector;


void main(void)
{
    //gl_FragColor = gl_Color;

    vec3 n,halfV;
    float NdotL,NdotHV;

    // store the ambient term
    vec4 color = ambient;

    // store the normalized interpolated normal
    n = normalize(normal);

    // compute the dot product between normal and ldir
    NdotL = max(dot(n,lightDir),0.0);
    if (NdotL > 0.0) {
        color += diffuse * NdotL;
        halfV = normalize(halfVector);
        NdotHV = max(dot(n,halfV),0.0);
        color += gl_FrontMaterial.specular * gl_LightSource[0].specular *
                pow(NdotHV, gl_FrontMaterial.shininess);
    }

    gl_FragColor = color;

    /*
    vec3 fogCol = vec3( 1.0, 1.0, 1.0);
    const float LOG2 = 1.442695;
    float fogDensity = 0.35;
    float z = gl_FragCoord.z / gl_FragCoord.w;
    float fogFactor = exp2( - fogDensity * fogDensity * z * z * LOG2 );
    fogFactor = clamp(fogFactor, 0.0, 1.0);
    gl_FragColor.rgb = mix( fogCol, gl_FragColor.rgb, fogFactor );
    */
}
