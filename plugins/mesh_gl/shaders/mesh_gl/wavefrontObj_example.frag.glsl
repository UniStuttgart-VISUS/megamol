#version 430

//#include "commondefines.glsl"

struct LightMetaInfo
{
    int point_light_cnt;
    int directional_light_cnt;
};

layout(std430, binding = 1) readonly buffer LightMetaInfoBuffer { LightMetaInfo light_meta_info; };

struct PointLights
{
    vec4 position_intensity;
};

layout(std430, binding = 2) readonly buffer PointLightsBuffer { PointLights[] point_lights; };

struct DirectionalLights
{
    vec4 direction_intensity;
};

layout(std430, binding = 3) readonly buffer DirectionalLightsBuffer { DirectionalLights[] directional_lights; };

layout(location = 0) in vec4 world_position;
layout(location = 1) in vec3 world_normal;

layout(location = 0) out vec4 frag_colour;

//Lambert Illumination 
float lambert(vec3 normal, vec3 light_dir)
{
    return clamp(dot(normal,light_dir),0.0,1.0);
}

void main(void) {

    vec4 retval = vec4(0.0,0.0,0.0,1.0);

    float depth = gl_FragCoord.z;
    if (depth > 0.0f && depth < 1.0f)
    {
        float reflected_light = 0.0;
        for(int i=0; i<light_meta_info.point_light_cnt; ++i)
        {
            vec3 light_dir = point_lights[i].position_intensity.xyz - world_position.xyz;
            float d = length(light_dir);
            light_dir = normalize(light_dir);
            reflected_light += lambert(world_normal,light_dir) * point_lights[i].position_intensity.w * (1.0/(d*d));
        }

        for(int i=0; i<light_meta_info.directional_light_cnt; ++i)
        {
            vec3 light_dir = -directional_lights[i].direction_intensity.xyz;
            reflected_light += lambert(world_normal,light_dir) * directional_lights[i].direction_intensity.w;
        }
        //Sets pixelcolor to illumination + color (alpha channels remains the same)
        retval.rgb = vec3(reflected_light);
    }

    frag_colour = retval;
}
