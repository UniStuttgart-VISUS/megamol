#version 450

uniform sampler2D normal_tx2D;
uniform sampler2D depth_tx2D;
uniform sampler2D noise_tx2D;

layout(OUTFORMAT) writeonly uniform image2D tgt_tx2D;


uniform mat4 view_mx;
uniform mat4 proj_mx;
uniform mat4 inv_view_mx;
uniform mat4 inv_proj_mx;

uniform float depth_threshold;
uniform float normal_threshold;


vec3 depthToViewPos(float depth, vec2 uv) {
    float z = depth * 2.0 - 1.0;

    vec4 cs_pos = vec4(uv * 2.0 - 1.0, z, 1.0);
    vec4 vs_pos = inv_proj_mx * cs_pos;

    // Perspective division
    vs_pos /= vs_pos.w;

    return vs_pos.xyz;
}


layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

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
    //normal = transpose(mat3(inv_view_mx)) * normal; // transform normal to view space

    float near = length(depthToViewPos(0.0,pixel_coords_norm));
    float far = length(depthToViewPos(1.0,pixel_coords_norm));
    float linear_depth = clamp(1.0 - ( (far - length(view_pos)) / (far - near)), 0.0, 1.0);

    //  float minSeparation = 1.0;
    //  float maxSeparation = 1.0;
    //  float minDistance   = 1.5;
    //  float maxDistance   = 2.0;
    //  float noiseScale    = 1.0;
    //  int   size          = 1;
    //  vec3  colorModifier = vec3(0.522, 0.431, 0.349);
    //  
    //  float separation = mix(maxSeparation, minSeparation, linear_depth);
    //  separation = 1.0;
    //  float count      = 1.0;
    //  float mx         = 0.0;
    //  
    //  for (int i = -size; i <= size; ++i) {
    //    for (int j = -size; j <= size; ++j) {
    //      vec2 texCoord = (vec2(i, j) * separation + (pixel_coords /* + noise */) + vec2(0.5)) / vec2(tgt_resolution);
    //      float sample_depth = texture( depth_tx2D, texCoord).r;
    //      vec3 sample_view_pos = depthToViewPos(sample_depth,texCoord);
    //      vec3 sample_normal = texture(normal_tx2D, texCoord).rgb;
    //  
    //      //if (sample_view_pos.z <= 0.0) {
    //      //    sample_view_pos.z = far;
    //      //}
    //  
    //      mx = max(mx, 1.0 - abs(dot(normal,sample_normal)));
    //  
    //      mx = max(mx, abs(length(view_pos) - length(sample_view_pos)));
    //      count += 1.0;
    //    }
    //  }
    //  
    //  float diff = smoothstep(minDistance, maxDistance, mx);

    vec3 west_sample_pos = depthToViewPos(texelFetchOffset( depth_tx2D, pixel_coords,0,ivec2(-1,0)).r,(vec2(-1,0) + (pixel_coords) + vec2(0.5)) / vec2(tgt_resolution));
    vec3 west_sample_normal = texelFetchOffset( normal_tx2D, pixel_coords,0,ivec2(-1,0)).rgb;
    vec3 east_sample_pos = depthToViewPos(texelFetchOffset( depth_tx2D, pixel_coords,0,ivec2(1,0)).r,(vec2(1,0) + (pixel_coords) + vec2(0.5)) / vec2(tgt_resolution));
    vec3 east_sample_normal = texelFetchOffset( normal_tx2D, pixel_coords,0,ivec2(1,0)).rgb;

    vec3 north_sample_pos = depthToViewPos(texelFetchOffset( depth_tx2D, pixel_coords,0,ivec2(0,1)).r,(vec2(0,1) + (pixel_coords) + vec2(0.5)) / vec2(tgt_resolution));
    vec3 north_sample_normal = texelFetchOffset( normal_tx2D, pixel_coords,0,ivec2(0,1)).rgb;
    vec3 south_sample_pos = depthToViewPos(texelFetchOffset( depth_tx2D, pixel_coords,0,ivec2(0,-1)).r,(vec2(0,-1) + (pixel_coords) + vec2(0.5)) / vec2(tgt_resolution));
    vec3 south_sample_normal = texelFetchOffset( normal_tx2D, pixel_coords,0,ivec2(0,-1)).rgb;

    int edge_detected = 0;
    edge_detected = abs(length(west_sample_pos) - length(east_sample_pos)) > depth_threshold ? edge_detected+1 : edge_detected;
    edge_detected = abs(length(north_sample_pos) - length(south_sample_pos)) > depth_threshold ? edge_detected+1 : edge_detected;

    edge_detected = (1.0 - abs(dot(west_sample_normal,east_sample_normal))) > normal_threshold ? edge_detected+1 : edge_detected;
    edge_detected = (1.0 - abs(dot(north_sample_normal,south_sample_normal))) > normal_threshold ? edge_detected+1 : edge_detected;

    imageStore(tgt_tx2D, pixel_coords, edge_detected > 0 ? vec4(0.0,0.0,0.0,1.0) : vec4(0.0,0.0,0.0,0.0));
}
