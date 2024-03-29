#version 120

varying vec4 diffuse,ambient;
varying vec3 normal,lightDir,halfVector;
varying vec4 col;

void main()
{
    // transformation of the normal into eye space
    normal = normalize(gl_NormalMatrix * gl_Normal);

    // normalize the direction of the light
    lightDir = normalize(vec3(gl_LightSource[0].position));

    // normalize the halfVector to pass it to the fragment shader
    halfVector = normalize(gl_LightSource[0].halfVector.xyz);

    // compute the diffuse, ambient and globalAmbient terms
    /*
    diffuse = gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse;
    ambient = gl_FrontMaterial.ambient * gl_LightSource[0].ambient;
    ambient += gl_LightModel.ambient * gl_FrontMaterial.ambient;
    */
    diffuse = gl_Color * gl_LightSource[0].diffuse;
    ambient = gl_Color * gl_LightSource[0].ambient;
    ambient += gl_LightModel.ambient * gl_Color;

    gl_Position = ftransform();
    gl_FrontColor = gl_Color;
    gl_BackColor = gl_Color;
}
