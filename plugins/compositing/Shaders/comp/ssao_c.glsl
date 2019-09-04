
struct Samples
{
    float x,y,z;
};

layout(std430, binding = 1) readonly buffer SamplesBuffer { Samples samples[]; };

uniform sampler2D normal_tx2D;
uniform sampler2D depth_tx2D;
uniform sampler2D noise_tx2D;

layout(RGBA16) writeonly uniform image2D tgt_tx2D;

uniform mat4 proj_mx;
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
    ivec2 tgt_resolution = imageSize (normal_tx2D);

    if (pixel_coords.x >= tgt_resolution.x || pixel_coords.y >= tgt_resolution.y) {
        return;
    }

    vec2 pixel_coords_norm = (vec2(pixel_coords) + vec2(0.5)) / vec2(tgt_resolution);

    // tile noise texture over screen based on screen dimensions divided by noise size
    vec2 noiseScale = vec2(tgt_resolution.x/4.0, tgt_resolution.y/4.0);
    
    vec3 normal    = texture(normal_tx2D, pixel_coords_norm).rgb;
    float depth    = texture(depth_tx2D, pixel_coords_norm).r
    vec3 rand_vec  = texture(texNoise, TexCoords * noiseScale).xyz;

    vec3 world_pos = depthToWorldPos(depth,pixel_coords_norm);

    vec3 tangent   = normalize(rand_vec - normal * dot(rand_vec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN       = mat3(tangent, bitangent, normal);


    float occlusion = 0.0;
    for(int i = 0; i < kernelSize; ++i)
    {
        // get sample position
        vec3 sample = TBN * samples[i]; // From tangent to view-space
        sample = fragPos + sample * radius; 

        vec4 offset = vec4(sample, 1.0);
        offset      = proj_mx * offset;       // from view to clip-space
        offset.xyz /= offset.w;               // perspective divide
        offset.xyz  = offset.xyz * 0.5 + 0.5; // transform to range 0.0 - 1.0

        float sample_depth = texture(depth_tx2D, offset.xy).z;
        float range_check = smoothstep(0.0, 1.0, radius / abs(fragPos.z - sample_depth));
        occlusion += (sample_depth >= sample.z + bias ? 1.0 : 0.0) * range_check;
    }

    imageStore(tgt_tx2D, pixel_coords , vec4(occlusion) );
}