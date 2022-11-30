#version 120
varying vec3 fragTangent;
varying vec3 fragView;

void main() {

    // Lighting params
    float k_ambient = 0.0;
    float k_diffuse = 0.8;
    float k_specular = 0.3;
    float k_exp = 50.0;

    vec3 t = normalize(fragTangent);
    float l_T = t.x*dot(normalize(gl_LightSource[0].position.xyz), t);
    float v_T = t.x*dot(normalize(-fragView.xyz), t);
    float diff = k_diffuse*sqrt(1.0-l_T*l_T);
    float spec = k_specular*pow(-v_T*l_T + sqrt(1.0-l_T*l_T)*sqrt(1.0-v_T*v_T), k_exp);
    gl_FragColor =  vec4(vec3(0.88, 0.86, 0.39)*(k_ambient + diff + spec), 1.0);
}
