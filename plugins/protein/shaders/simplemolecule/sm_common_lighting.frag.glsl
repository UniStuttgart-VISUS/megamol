#version 430

struct LightParams
{
    float x,y,z,intensity;
};

layout(std430, binding = 1) readonly buffer PointLightParamsBuffer { LightParams point_light_params[]; };
layout(std430, binding = 2) readonly buffer DistantLightParamsBuffer { LightParams distant_light_params[]; };

uniform int point_light_cnt;
uniform int distant_light_cnt;

uniform sampler2D albedo_tx2D;
uniform sampler2D normal_tx2D;
uniform sampler2D depth_tx2D;

uniform vec4 ambientColor;
uniform vec4 diffuseColor;
uniform vec4 specularColor;

uniform float k_amb;
uniform float k_diff;
uniform float k_spec;
uniform float k_exp;

in vec2 uv_coord;

vec3 blinnPhong(vec3 normal, vec3 lightdirection, vec3 v){
    vec3 Colorout;

    //Ambient Part
    vec3 Camb = k_amb * ambientColor.rgb;

    //Diffuse Part
    vec3 Cdiff = diffuseColor.rgb * k_diff * clamp(dot(normal,lightdirection),0,1);

    //Specular Part
    vec3 h = normalize(v + lightdirection);
    normal = normal / sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
    float costheta = clamp(dot(h,normal),0,1);
    vec3 Cspek = specularColor.rgb * k_spec * ((k_exp + 2)/(2 * 3.141592f)) * pow(costheta, k_exp);

    //Final Equation
    Colorout = Camb + Cdiff + Cspek;
    return Colorout;
}

float lambert(vec3 normal, vec3 light_dir)
{
    return clamp(dot(normal,light_dir),0.0,1.0);
}

vec3 depthToWorldPos(float depth, vec2 uv, mat4 invview, mat4 invproj) {
    float z = depth * 2.0 - 1.0;

    vec4 cs_pos = vec4(uv * 2.0 - 1.0, z, 1.0);
    vec4 vs_pos = invproj * cs_pos;

    // Perspective division
    vs_pos /= vs_pos.w;
    
    vec4 ws_pos = invview * vs_pos;

    return ws_pos.xyz;
}

void main(void) {
    vec4 albedo = texture(albedo_tx2D, uv_coord);
    vec3 normal = texture(normal_tx2D, uv_coord).rgb;
    float depth = texture(depth_tx2D, uv_coord).r; 

    // TODO implement

    gl_FragColor = vec4(albedo.xyz, 1);
}