#version 430

#include "mmstd_gl/shading/color.inc.glsl"
#include "mmstd_gl/shading/tonemapping.inc.glsl"
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

layout(OUTFORMAT) writeonly uniform image2D tgt_tx2D;

uniform int point_light_cnt;
uniform int distant_light_cnt;
uniform vec3 camPos;

uniform mat4 inv_view_mx;
uniform mat4 inv_proj_mx;

uniform float exposure_avg_intensity;
uniform float roughness;

struct LightIntensities{
    float diffuse;
    float specular;
    float rim;
    float sheen;
};

LightIntensities computeLightTerms(vec3 n, vec3 l, vec3 v, float spec_exp){
    LightIntensities retval;

    float n_dot_l = clamp(dot(n,l),0.0,1.0);
    vec3 h = normalize(l+v);
    float n_dot_h = clamp(dot(n,h),0.0,1.0);
    vec3 r = reflect(-l, n);
    float r_dot_v = clamp(dot(r, v), 0.0,1.0);

    retval.diffuse = n_dot_l; 
    retval.specular = ((spec_exp + 2.0)/(2.0 * 3.141592f)) * pow(r_dot_v, spec_exp);


    float v_dot_n_inv = pow(1.0 - abs(dot(v,n)),5.0);
    retval.rim = v_dot_n_inv * smoothstep(0.0,0.2,n_dot_l);

    return retval;
}

float toonRamp(float light_intensity, float shadow, float midtone, float light){
    float feathering = 0.005;

    // note: use midtone and light as threshold and intensity, but set shadow always to black
    float shadow_to_midtone = mix(0.0,midtone,smoothstep(shadow,shadow + feathering, light_intensity));
    float midtone_to_light = mix(midtone,light,smoothstep(midtone,midtone + feathering, light_intensity));

    return light_intensity > midtone ? midtone_to_light : shadow_to_midtone;
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

    vec4 retval = albedo;

    if (depth > 0.0f && depth < 1.0f)
    {
        vec3 world_pos = depthToWorldPos(depth,pixel_coords_norm, inv_view_mx, inv_proj_mx);

        LightIntensities reflected_light;
        reflected_light.diffuse = 0.0;
        reflected_light.specular = 0.0;
        reflected_light.rim = 0.0;

        // gather diffuse and specular reflections seperatley 
        for(int i=0; i<point_light_cnt; ++i)
        {
            vec3 light_dir = vec3(point_light_params[i].x,point_light_params[i].y,point_light_params[i].z) - world_pos;
            float d = length(light_dir);
            light_dir = normalize(light_dir);
            vec3 view_dir = normalize(camPos - world_pos);
            LightIntensities li = computeLightTerms(normal,light_dir, view_dir, mix(100,1,roughness));
            reflected_light.diffuse += li.diffuse * point_light_params[i].intensity * (1.0/(d*d));
            reflected_light.specular += li.specular * point_light_params[i].intensity * (1.0/(d*d));
            reflected_light.rim += li.rim * point_light_params[i].intensity * (1.0/(d*d));
        }
        
        for(int i=0; i<distant_light_cnt; ++i)
        {
            vec3 light_dir = -1.0 * vec3(distant_light_params[i].x,distant_light_params[i].y,distant_light_params[i].z);
            vec3 view_dir = normalize(camPos - world_pos);
            LightIntensities li = computeLightTerms(normal,light_dir, view_dir, mix(42,1,roughness));
            reflected_light.diffuse += li.diffuse * distant_light_params[i].intensity;
            reflected_light.specular += li.specular * distant_light_params[i].intensity;
            reflected_light.rim += li.rim * distant_light_params[i].intensity;
        }

        // apply exposure, i.e. map avg intensity to 0.18; 0.18 as target for avg intensity comes back to gamme correction
        float exposure_value = (0.18/exposure_avg_intensity);
        float c_white = (1.0/exposure_value);
        reflected_light.diffuse *= exposure_value;
        reflected_light.specular *= exposure_value;
        reflected_light.rim *= exposure_value;

        //////
        // Option 1:
        // Tone map before toon ramp to get a stable light value (max intensity = 1)
        // Note: diffuse and specular midtone and light are the same due to compressed
        // range after tone mapping
        //
        //  // compress to range [0,1] using Reihard operator
        //  reflected_light.diffuse = reinhardExt(reflected_light.diffuse,c_white);
        //  reflected_light.specular = reinhardExt(reflected_light.specular,c_white);
        //  reflected_light.rim = reinhardExt(reflected_light.rim,c_white);
        //  // apply toon color ramp
        //  float diffuse_shadow = 0.0;
        //  float diffuse_midtone = reinhardExt(0.18,c_white);
        //  float diffuse_light = 1.0;
        //  float specular_shadow = reinhardExt(0.0,c_white);
        //  float specular_midtone = reinhardExt(0.18,c_white);
        //  float specular_light = 1.0;
        //  float rim_shadow = reinhardExt(0.045,c_white);
        //  float rim_midtone = reinhardExt(0.09,c_white);
        //  float rim_light = 1.0;

        //////
        // Option 2:
        // Apply toon ramp first, choose some stable light value (e.g. 1).
        // By doing so, toon ramp will clamp the dynamic range, therefore
        // use that value afterwards as c_white for tone mapping to retain full
        // range in LDR. Note: Diffuse and specular midtone and light value
        // are set different because toon ramp is applied to full dynamic range
        //
        // apply toon color ramp
        float diffuse_shadow = 0.02;
        float diffuse_midtone = 0.18;
        float diffuse_light = 1.0;
        float specular_shadow = 0.18;
        float specular_midtone = 0.36;
        float specular_light = 1.0;
        float rim_shadow = 0.045;
        float rim_midtone = 0.09;
        float rim_light = 1.0;

        reflected_light.diffuse = toonRamp(reflected_light.diffuse, diffuse_shadow, diffuse_midtone, diffuse_light);
        reflected_light.specular = toonRamp(reflected_light.specular, specular_shadow, specular_midtone, specular_light);
        reflected_light.rim = toonRamp(reflected_light.rim, rim_shadow, rim_midtone, rim_light);

        // combines specular and rim lighting with max operator, then balance with diffuse based on roughness
        float specular = max(reflected_light.specular, reflected_light.rim);
        vec3 final_light_intensity =vec3(mix(specular, reflected_light.diffuse, roughness));

        // apply tone mapping and gamma correction
        final_light_intensity = toSRGB(vec4(reinhardExt(final_light_intensity,vec3(1.0)),1.0)).rgb;

        // multiply sRGB color with gamma corrected light intensity
        retval.rgb = (albedo.rgb * final_light_intensity);
    }

    imageStore(tgt_tx2D, pixel_coords , retval );
}
