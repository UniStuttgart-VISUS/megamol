varying vec4 diffuse,ambient;
varying vec3 normal,lightDir,halfVector;

varying vec4 posWS;



void main()
{    

    posWS = gl_Vertex;
    // transformation of the normal into eye space
    normal = normalize(gl_NormalMatrix * gl_Normal);
    
    // normalize the direction of the light
    lightDir = normalize(vec3(gl_LightSource[0].position));

    // normalize the halfVector to pass it to the fragment shader
    halfVector = normalize(gl_LightSource[0].halfVector.xyz);
    
    vec4 tmp = gl_ModelViewMatrix * gl_ModelViewMatrixInverse[2];
    vec3 camIn = normalize(tmp.xyz);
    //if(dot(normal, lightDir) < 0.0) normal *= -1.0;
    //if(dot(normal, halfVector) < 0.0) normal *= -1.0;
    if(dot(normal, camIn) < 0.0) normal *= -1.0;
                
    // compute the diffuse, ambient and globalAmbient terms
    /*
    diffuse = gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse;
    ambient = gl_FrontMaterial.ambient * gl_LightSource[0].ambient;
    ambient += gl_LightModel.ambient * gl_FrontMaterial.ambient;
    */
    //vec4 col = vec4(0.5, 0.8, 0.2, 1.0);
    //vec4 col = vec4(0.0, 0.8, 0.8, 1.0);
    vec4 col = vec4(1.0, 0.75, 0.0, 1.0);
    /*diffuse = gl_Color * gl_LightSource[0].diffuse;
    ambient = gl_Color * gl_LightSource[0].ambient;
    ambient += gl_LightModel.ambient * gl_Color;*/
    diffuse = col * gl_LightSource[0].diffuse;
    ambient = col * gl_LightSource[0].ambient;
    ambient += gl_LightModel.ambient * col;
    
    // if(dot(normal, camIn) < 0.0) 
        // ambient = vec4(1.0, 0.0, 0.0, 1.0);
    // else 
        // ambient = vec4(0.0, 1.0, 0.0, 1.0);
        
    gl_Position = ftransform();
    
    gl_TexCoord[0] = gl_MultiTexCoord0; // 3D tex coords
}
