#version 430

struct LightParams {
    float x,y,z,intensity;
};

layout(std430, binding = 1) readonly buffer PointLightParamsBuffer { LightParams point_light_params[]; };
layout(std430, binding = 2) readonly buffer DistantLightParamsBuffer { LightParams distant_light_params[]; };

layout(location = 0) out vec4 albedo_out;
layout(location = 1) out vec3 normal_out;
layout(location = 2) out float depth_out;

in vec3 pass_pos;
in vec3 pass_normal;
in vec3 pass_color;
in vec2 pass_texcoord;

uniform int point_light_cnt = 0;
uniform int distant_light_cnt = 0;

uniform vec4 ambientColor = vec4(1);
uniform vec4 diffuseColor = vec4(1);
uniform vec4 specularColor = vec4(1);
uniform vec3 meshColor = vec3(1);

uniform float k_amb = 0.2;
uniform float k_diff = 0.7;
uniform float k_spec = 0.1;
uniform float k_exp = 120.0;

uniform bool use_lambert = false;
uniform bool enable_lighting = true;
uniform bool has_normals = true;
uniform bool has_colors = true;
uniform bool has_texcoords = false;

uniform vec3 cam_pos;

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

float lambert(vec3 normal, vec3 light_dir) {
    return clamp(dot(normal,light_dir),0.0,1.0);
}

void main() {
    albedo_out = vec4(pass_color, 1.0);
    normal_out = normalize(pass_normal);
    depth_out = gl_FragCoord.z;

    if(!has_colors) {
        albedo_out = vec4(meshColor, 1.0);
    }

    bool has_lights = point_light_cnt > 0 || distant_light_cnt > 0;
    if(enable_lighting && has_lights && has_normals) {
        vec4 retval = albedo_out;
        vec3 normal = normalize(pass_normal);
        vec3 reflected_light = vec3(0.0);
        for(int i = 0; i < point_light_cnt; ++i) {
            vec3 light_dir = vec3(point_light_params[i].x, point_light_params[i].y, point_light_params[i].z) - pass_pos;
            float d = length(light_dir);
            light_dir = normalize(light_dir);
            vec3 view_dir = normalize(cam_pos - pass_pos);
            if(use_lambert) {
                reflected_light += lambert(normal, light_dir);
            } else {
                reflected_light += blinnPhong(normal, light_dir, view_dir, ambientColor, diffuseColor, specularColor, vec4(k_amb, k_diff, k_spec, k_exp)) * point_light_params[i].intensity * (1.0/(d*d));
            }
        }
        for(int i = 0; i < distant_light_cnt; ++i) {
            vec3 light_dir = -1.0 * vec3(distant_light_params[i].x, distant_light_params[i].y, distant_light_params[i].z);
            vec3 view_dir = normalize(cam_pos - pass_pos);
            if(use_lambert){
                reflected_light += lambert(normal, light_dir);
            }else {
                reflected_light += blinnPhong(normal, light_dir, view_dir, ambientColor, diffuseColor, specularColor, vec4(k_amb, k_diff, k_spec, k_exp)) * distant_light_params[i].intensity;
            }
        }
        retval.rgb = reflected_light * albedo_out.rgb;
        albedo_out = vec4(retval.rgb, albedo_out.a);
    }
}
