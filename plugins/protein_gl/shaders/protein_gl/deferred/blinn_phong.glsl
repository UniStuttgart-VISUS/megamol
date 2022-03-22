vec3 blinnPhong(vec3 normal, vec3 lightdirection, vec3 v, vec4 ambientCol, vec4 diffuseCol, vec4 specCol, vec4 params){
    vec3 Colorout;

    //Ambient Part
    vec3 Camb = params.x * ambientCol.rgb;

    //Diffuse Part
    vec3 Cdiff = diffuseCol.rgb * params.y * clamp(dot(normal,lightdirection),0,1);

    //Specular Part
    vec3 h = normalize(v + lightdirection);
    normal = normal / sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
    float costheta = clamp(dot(h,normal),0,1);
    vec3 Cspek = specCol.rgb * params.z * ((params.w + 2)/(2 * 3.141592f)) * pow(costheta, params.w);

    //Final Equation
    Colorout = Camb + Cdiff + Cspek;
    return Colorout;
}
