#version 120

uniform int twoSidedLight = 0;

varying vec4 diffuse,ambient;
varying vec3 normal,lightDir,halfVector;


void main() {
    float tsl = clamp(float(twoSidedLight), 0.0, 1.0);
    vec3 n,halfV;
    float NdotL,NdotHV;

    // store the ambient term
    vec4 color = ambient;

    // store the normalized interpolated normal
    n = normalize(normal);

    // compute the dot product between normal and lightDir
    NdotL = dot(n,lightDir);
    if (NdotL > 0.0) {
        // front side
        color += diffuse * NdotL;
        halfV = normalize(halfVector);
        NdotHV = max(dot(n,halfV),4.88e-04);
        color += gl_FrontMaterial.specular * gl_LightSource[0].specular * pow(NdotHV, gl_FrontMaterial.shininess);
    } else {
        // back side
        color += diffuse * (-NdotL) * tsl;
        //halfV = normalize(halfVector);
        //NdotHV = max(dot(-n,halfV),4.88e-04);
        //color += gl_FrontMaterial.specular * gl_LightSource[0].specular * pow(NdotHV, gl_FrontMaterial.shininess) * tsl;
    }

    gl_FragColor = color;
}
