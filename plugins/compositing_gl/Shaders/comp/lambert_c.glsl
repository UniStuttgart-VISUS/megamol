struct LightParams
{
    float x,y,z,intensity;
};

layout(std430, binding = 1) readonly buffer PointLightParamsBuffer { LightParams point_light_params[]; };
layout(std430, binding = 2) readonly buffer DistantLightParamsBuffer { LightParams distant_light_params[]; };

uniform sampler2D albedo_tx2D;
uniform sampler2D normal_tx2D;
uniform sampler2D depth_tx2D;

layout(RGBA16) writeonly uniform image2D tgt_tx2D;

uniform int point_light_cnt;
uniform int distant_light_cnt;

uniform mat4 inv_view_mx;
uniform mat4 inv_proj_mx;


vec3 depthToWorldPos(float depth, vec2 uv) {
    float z = depth * 2.0 - 1.0;

    vec4 cs_pos = vec4(uv * 2.0 - 1.0, z, 1.0);
    vec4 vs_pos = inv_proj_mx * cs_pos;

    // Perspective division
    vs_pos /= vs_pos.w;
    
    vec4 ws_pos = inv_view_mx * vs_pos;

    return ws_pos.xyz;
}

float lambert(vec3 normal, vec3 light_dir)
{
    return clamp(dot(normal,light_dir),0.0,1.0);
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
        vec3 world_pos = depthToWorldPos(depth,pixel_coords_norm);

        float reflected_light = 0.0;
        for(int i=0; i<point_light_cnt; ++i)
        {
            vec3 light_dir = vec3(point_light_params[i].x,point_light_params[i].y,point_light_params[i].z) - world_pos;
            float d = length(light_dir);
            light_dir = normalize(light_dir);
            reflected_light += lambert(light_dir,normal) * point_light_params[i].intensity * (1.0/(d*d));
        }

        for(int i=0; i<distant_light_cnt; ++i)
        {
            vec3 light_dir = vec3(distant_light_params[i].x,distant_light_params[i].y,distant_light_params[i].z);
            reflected_light += lambert(light_dir,normal) * distant_light_params[i].intensity;
        }

        retval.rgb = vec3(reflected_light) * albedo.rgb;
    }

    imageStore(tgt_tx2D, pixel_coords , retval );
}