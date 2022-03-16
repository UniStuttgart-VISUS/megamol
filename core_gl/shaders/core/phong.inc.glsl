#ifndef CORE_PHONG_INC_GLSL
#define CORE_PHONG_INC_GLSL

uniform float ka;
uniform float kd;
uniform float ks;
uniform float shininess;

uniform vec3 light;

uniform vec3 ambient_col;
uniform vec3 specular_col;
uniform vec3 light_col;

vec3 phong(vec3 color, vec3 normal, vec3 eye, vec3 light) {
    eye = normalize(eye);
    light = normalize(light);

    vec3 ambient = ka * ambient_col * color;
    vec3 diffuse = kd * light_col * color * max(0, dot(normal, light));
    vec3 specular = ks * light_col * specular_col * pow(max(0, dot(normal, normalize(light + eye))), shininess);

    return clamp(ambient + diffuse + specular, vec3(0), vec3(1));
}

#endif // CORE_PHONG_INC_GLSL
