// see https://64.github.io/tonemapping/
vec3 reinhardExt(vec3 c, vec3 c_white){
    return (c * (vec3(1.0)+(c/pow(c_white,vec3(2.0)))))/(vec3(1.0)+c);
}
float reinhardExt(float c, float c_white){
    return (c * (1.0+(c/pow(c_white,2.0))))/(1.0+c);
}
