// normal: Normal in object space
// col:    Color
vec4 DirectLight(vec3 normal, vec4 col) {
  float nDotVP;
  float nDotHV;
  float pf;
  vec4 tempColor = vec4(0.f);

  // OLD normal = (gl_ModelViewMatrixInverseTranspose * vec4(normal, 1.0)).xyz;
  normal = (ModelViewMatrixInverseTranspose * vec4(normal, 1.0)).xyz;
  normal = normalize(normal);

  for(int i = 0; i < numLights; ++i) {
    // OLD nDotVP = max(0.0, dot(normal, normalize(gl_LightSource[0].position.xyz)));
    nDotVP = max(0.0, dot(normal, normalize(vec3(light[i].px, light[i].py, light[i].pz))));
    //nDotHV = max(0.0, dot(normal, gl_LightSource[0].halfVector.xyz));

    //if (nDotVP == 0.0) {
    //  pf = 0.0;
    //} else {
    //  pf = pow(nDotHV, gl_FrontMaterial.shininess);
    //}

    // OLD vec4 ambient = gl_LightSource[0].ambient;
    // OLD vec4 diffuse = gl_LightSource[0].diffuse * nDotVP;
    // OLD //vec4 specular = gl_LightSource[0].specular * pf;
    vec4 ambient = light[i].lightIntensity * ambientCol;
    vec4 diffuse = light[i].lightIntensity * diffuseCol * nDotVP;
    //vec4 specular = specularCol * pf;

    tempColor += vec4(col.rgb * ambient.rgb + col.rgb * diffuse.rgb /*+ specular.rgb*/, 1.0);
  }

  return tempColor;
}
