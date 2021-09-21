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

uniform mat4 inv_view_mx;
uniform mat4 inv_proj_mx;
uniform vec3 camPos;

uniform vec4 ambientColor = vec4(1);
uniform vec4 diffuseColor = vec4(1);
uniform vec4 specularColor = vec4(1);

uniform float k_amb = 0.2;
uniform float k_diff = 0.7;
uniform float k_spec = 0.1;
uniform float k_exp = 120.0;

uniform bool use_lambert = false;

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

    vec4 retval = albedo;

    if(depth > 0.0f && depth < 1.0f) {
        vec3 world_pos = depthToWorldPos(depth, uv_coord, inv_view_mx, inv_proj_mx);
        vec3 reflected_light = vec3(0.0);
        for(int i = 0; i < point_light_cnt; ++i) {
            vec3 light_dir = vec3(point_light_params[i].x, point_light_params[i].y, point_light_params[i].z) - world_pos;
            float d = length(light_dir);
            light_dir = normalize(light_dir);
            vec3 view_dir = normalize(camPos - world_pos);
            if(use_lambert) {
                reflected_light += lambert(normal, light_dir);
            } else {
                reflected_light += blinnPhong(normal, light_dir, view_dir) * point_light_params[i].intensity * (1.0/(d*d));
            }
        }
        for(int i = 0; i < distant_light_cnt; ++i) {
            vec3 light_dir = -1.0 * vec3(distant_light_params[i].x, distant_light_params[i].y, distant_light_params[i].z);
            vec3 view_dir = normalize(camPos - world_pos);
            if(use_lambert){
                reflected_light += lambert(normal, light_dir);
            }else {
                reflected_light += blinnPhong(normal, light_dir, view_dir) * distant_light_params[i].intensity;
            }
        }
        retval.rgb = reflected_light * albedo.rgb;
    }

    gl_FragColor = vec4(retval.xyz, 1);
    gl_FragDepth = depth;

    if(albedo.w == 0.0) {
        discard;
    }
}