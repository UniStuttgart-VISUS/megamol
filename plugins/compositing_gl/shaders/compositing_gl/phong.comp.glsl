#version 430

#include "mmstd_gl/shading/color.inc.glsl"
#include "mmstd_gl/shading/transformations.inc.glsl"

struct LightParams
{
    float x,y,z,intensity;
};

layout(std430, binding = 1) readonly buffer PointLightParamsBuffer { LightParams point_light_params[]; };
layout(std430, binding = 2) readonly buffer DistantLightParamsBuffer { LightParams distant_light_params[]; };

uniform sampler2D albedo_tx2D;
uniform sampler2D normal_tx2D;
uniform sampler2D depth_tx2D;

layout(rgba16) writeonly uniform image2D tgt_tx2D;

uniform int point_light_cnt;
uniform int distant_light_cnt;
uniform vec3 camPos;

uniform mat4 inv_view_mx;
uniform mat4 inv_proj_mx;

//ambient light for ambientcolor todo
uniform vec4 ambientColor;
//textur bezogen for diffcolor todo
uniform vec4 diffuseColor;
uniform vec4 specularColor;

uniform float k_amb;
uniform float k_diff;
uniform float k_spec;
uniform float k_exp;

//TODO M_PI

//TODO: ambient part adding via light component through uniforms like Point/Distant lights

//Blinn-Phong Illumination 
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

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    uvec3 gID = gl_GlobalInvocationID.xyz;
    ivec2 pixel_coords = ivec2(gID.xy);
    ivec2 tgt_resolution = imageSize (tgt_tx2D);

    if (pixel_coords.x >= tgt_resolution.x || pixel_coords.y >= tgt_resolution.y) {
        return;
    }

    vec2 pixel_coords_norm = (vec2(pixel_coords) + vec2(0.5)) / vec2(tgt_resolution);

    vec4 albedo = texture(albedo_tx2D,pixel_coords_norm);
    vec3 normal = texture(normal_tx2D,pixel_coords_norm).rgb;
    float depth = texture(depth_tx2D,pixel_coords_norm).r;

    //var for saving alpha channel 
    vec4 retval = albedo;

    if (depth > 0.0f && depth < 1.0f)
    {
        vec3 world_pos = depthToWorldPos(depth,pixel_coords_norm,inv_view_mx,inv_proj_mx);

        vec3 reflected_light = vec3(0.0);
        for(int i=0; i<point_light_cnt; ++i)
        {
            vec3 light_dir = vec3(point_light_params[i].x,point_light_params[i].y,point_light_params[i].z) - world_pos;
            float d = length(light_dir);
            light_dir = normalize(light_dir);
            vec3 view_dir = normalize(camPos - world_pos);
            reflected_light += blinnPhong(normal,light_dir, view_dir) * point_light_params[i].intensity * (1.0/(d*d));
       
        }
        
        for(int i=0; i<distant_light_cnt; ++i)
        {
            vec3 light_dir = -1.0 * vec3(distant_light_params[i].x,distant_light_params[i].y,distant_light_params[i].z);
            vec3 view_dir = normalize(camPos - world_pos);
            reflected_light += blinnPhong(normal,light_dir, view_dir) * distant_light_params[i].intensity;
        
        }
        //Sets pixelcolor to illumination + color (alpha channels remains the same)
        //albedo = ambi/diff koeff auf albedo 
        retval.rgb = toSRGB(vec4(reflected_light,1.0)).rgb * albedo.rgb;
    }

    imageStore(tgt_tx2D, pixel_coords , retval );
}
