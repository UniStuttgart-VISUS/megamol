#version 430

struct LightParams
{
    float x,y,z,intensity;
};

layout(std430, binding = 1) readonly buffer PointLightParamsBuffer { LightParams point_light_params[]; };
layout(std430, binding = 2) readonly buffer DistantLightParamsBuffer { LightParams distant_light_params[]; };

layout(location = 0) out vec4 color_out;

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
uniform bool no_lighting = false;

in vec2 uv_coord;

#include "mmstd_gl/shading/transformations.inc.glsl"
#include "protein_gl/deferred/blinn_phong.glsl"
#include "protein_gl/deferred/lambert.glsl"

void main(void) {
    vec4 albedo = texture(albedo_tx2D, uv_coord);
    vec3 normal = texture(normal_tx2D, uv_coord).rgb;
    float depth = texture(depth_tx2D, uv_coord).r; 

    vec4 retval = albedo;

    gl_FragDepth = depth;

    if(albedo.w < 0.001) {
        discard;
    }

    if(no_lighting) {
        color_out = albedo;
        return;
    }

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
                reflected_light += blinnPhong(normal, light_dir, view_dir, ambientColor, diffuseColor, specularColor, vec4(k_amb, k_diff, k_spec, k_exp)) * point_light_params[i].intensity * (1.0/(d*d));
            }
        }
        for(int i = 0; i < distant_light_cnt; ++i) {
            vec3 light_dir = -1.0 * vec3(distant_light_params[i].x, distant_light_params[i].y, distant_light_params[i].z);
            vec3 view_dir = normalize(camPos - world_pos);
            if(use_lambert){
                reflected_light += lambert(normal, light_dir);
            } else {
                reflected_light += blinnPhong(normal, light_dir, view_dir, ambientColor, diffuseColor, specularColor, vec4(k_amb, k_diff, k_spec, k_exp)) * distant_light_params[i].intensity;
            }
        }
        retval.rgb = reflected_light * albedo.rgb;
    }

    color_out = vec4(retval.xyz, albedo.w);
}
