
struct Samples
{
    float x,y,z;
};

layout(std430, binding = 1) readonly buffer SamplesBuffer { Samples samples[]; };
uniform int sample_cnt;
uniform float radius;

uniform sampler2D normal_tx2D;
uniform sampler2D depth_tx2D;
uniform sampler2D noise_tx2D;

layout(RGBA16) writeonly uniform image2D tgt_tx2D;

uniform mat4 view_mx;
uniform mat4 proj_mx;
uniform mat4 inv_view_mx;
uniform mat4 inv_proj_mx;


vec3 depthToViewPos(float depth, vec2 uv) {
    float z = depth * 2.0 - 1.0;

    vec4 cs_pos = vec4(uv * 2.0 - 1.0, z, 1.0);
    vec4 vs_pos = inv_proj_mx * cs_pos;

    // Perspective division
    vs_pos /= vs_pos.w;

    return vs_pos.xyz;
}


layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// see https://learnopengl.com/Advanced-Lighting/SSAO
void main()
{
    uvec3 gID = gl_GlobalInvocationID.xyz;
    ivec2 pixel_coords = ivec2(gID.xy);
    ivec2 tgt_resolution = imageSize (tgt_tx2D);

    if (pixel_coords.x >= tgt_resolution.x || pixel_coords.y >= tgt_resolution.y) {
        return;
    }

    vec2 pixel_coords_norm = (vec2(pixel_coords) + vec2(0.5)) / vec2(tgt_resolution);

    // tile noise texture over screen based on screen dimensions divided by noise size
    vec2 noise_scale = vec2(tgt_resolution.x/4.0, tgt_resolution.y/4.0);
    
    vec3 normal    = texture(normal_tx2D, pixel_coords_norm).rgb;
    float depth    = texture(depth_tx2D, pixel_coords_norm).r;
    vec3 rand_vec  = texture(noise_tx2D, pixel_coords_norm * noise_scale).xyz;

    vec3 view_pos = depthToViewPos(depth,pixel_coords_norm);

    normal = transpose(mat3(inv_view_mx)) * normal; // transform normal to view space
    vec3 tangent    = normalize(rand_vec - normal * dot(rand_vec, normal));
    vec3 bitangent  = cross(normal, tangent);
    mat3 tangent_mx = mat3(tangent, bitangent, normal);
    
    float bias = 0.0001;

    float occlusion = 0.0;

    vec3 sample_vs_pos;
    vec3 frag_vs_pos;

    if(depth > 0.0)
    {
        for(int i = 0; i < sample_cnt; ++i)
        {
            // get sample position
            sample_vs_pos = tangent_mx * vec3(samples[i].x,samples[i].y,samples[i].z); // From tangent to view-space
            sample_vs_pos = view_pos + sample_vs_pos * radius; 

            vec4 offset = vec4(sample_vs_pos, 1.0);
            offset      = proj_mx * offset;       // from view to clip-space
            offset.xyz /= offset.w;               // perspective divide
            offset.xyz  = offset.xyz * 0.5 + 0.5; // transform to range 0.0 - 1.0

            if(offset.x > 1.0 || offset.x < 0.0 || offset.y > 1.0 || offset.y < 0.0f)
                continue;

            float sample_depth = texture(depth_tx2D, offset.xy).r;

            if(sample_depth < 0.0001)
                continue;

            frag_vs_pos = depthToViewPos(sample_depth,offset.xy);

            float range_check = smoothstep(0.0, 1.0, radius / abs(length(view_pos) - length(frag_vs_pos))); 
            occlusion += (length(frag_vs_pos) <= length(sample_vs_pos) - bias ? 1.0 : 0.0) * range_check; 
        }
    }
    
    occlusion = 1.0 - (occlusion / sample_cnt);

    imageStore(tgt_tx2D, pixel_coords, vec4(occlusion, occlusion, occlusion, 1.0));
}