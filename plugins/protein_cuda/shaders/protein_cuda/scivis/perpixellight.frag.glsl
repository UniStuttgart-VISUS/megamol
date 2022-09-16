varying vec4 diffuse,ambient;
varying vec3 normal,lightDir,halfVector;
varying vec4 posWS;

uniform float posXMax;
uniform float posYMax;
uniform float posZMax;

uniform float posXMin;
uniform float posYMin;
uniform float posZMin;

uniform sampler3D curlMagTex;

void main()
{
    if(posWS.x > posXMax) discard;
    if(posWS.y > posYMax) discard;
    if(posWS.z > posZMax) discard;

    if(posWS.x < posXMin) discard;
    if(posWS.y < posYMin) discard;
    if(posWS.z < posZMin) discard;

    /*if(posWS.x > 93.0) discard;
    if(posWS.y > 93.0) discard;
    if(posWS.z > 93.0) discard;

    if(posWS.x < -93.0) discard;
    if(posWS.y < -93.0) discard;
    if(posWS.z < -93.0) discard;*/

    // For height ridge zoom
    /*if(posWS.x > -19.279999) discard;
    if(posWS.y > -1.84) discard;
    if(posWS.z > 93.050003) discard;

    if(posWS.x < -47.180000) discard;
    if(posWS.y < -35.000000) discard;
    if(posWS.z < 84.919998) discard;*/

    vec3 n,halfV;
    float NdotL,NdotHV;

    // store the ambient term
    vec4 color = ambient;

    // store the normalized interpolated normal
    n = normalize(normal);

    // compute the dot product between normal and lightDir
    NdotL = max(dot(n,lightDir),0.0);
    if (NdotL > 0.0)
    {
        color += diffuse * NdotL;
        halfV = normalize(halfVector);
        NdotHV = max(dot(n,halfV),4.88e-04);
        color += gl_FrontMaterial.specular * gl_LightSource[0].specular * pow(NdotHV, gl_FrontMaterial.shininess);
    }

    gl_FragColor = color;
    //gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0);
    //gl_FragColor = vec4(gl_TexCoord[0].stp, 1.0);
    //gl_FragColor = vec4(vec3(texture3D(curlMagTex, gl_TexCoord[0].stp).a*5.0), 1.0); // Color by curl magnitude
}
