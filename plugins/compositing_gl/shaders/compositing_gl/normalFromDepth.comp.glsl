#version 430

uniform sampler2D src_tx2D;

layout(RGBA16) writeonly uniform image2D tgt_tx2D;

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

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main()
{
    uvec3 gID = gl_GlobalInvocationID.xyz;
    ivec2 pixel_coords = ivec2(gID.xy);
    ivec2 tgt_resolution = imageSize(tgt_tx2D);

    if (pixel_coords.x >= tgt_resolution.x || pixel_coords.y >= tgt_resolution.y) {
        return;
    }

    vec2 pixel_coords_norm = (vec2(pixel_coords) + vec2(0.5)) / vec2(tgt_resolution);
    vec2 pixel_offset = vec2(1.0) / vec2(tgt_resolution);

    float depth_c = texture(src_tx2D,pixel_coords_norm).r;
    float depth_t = texture(src_tx2D,pixel_coords_norm + vec2(0.0, pixel_offset.y)).r;
    float depth_b = texture(src_tx2D,pixel_coords_norm - vec2(0.0, pixel_offset.y)).r;
    float depth_l = texture(src_tx2D,pixel_coords_norm - vec2(pixel_offset.x, 0.0)).r;
    float depth_r = texture(src_tx2D,pixel_coords_norm + vec2(pixel_offset.x, 0.0)).r;
    

    if ((depth_c > 0.0f) && (depth_c < 1.0f)){
        vec3 position_c = depthToWorldPos(depth_c,pixel_coords_norm);

        vec3 position_bt = ((depth_t) > 0.0f && (depth_t < 1.0f)) ? depthToWorldPos(depth_t,pixel_coords_norm + vec2(0.0, pixel_offset.y)) :
            ((depth_b > 0.0f) && (depth_b < 1.0f)) ? depthToWorldPos(depth_b,pixel_coords_norm - vec2(0.0, pixel_offset.y)) : position_c;

        vec3 position_t = ((depth_r > 0.0f) && (depth_r < 1.0f)) ? depthToWorldPos(depth_r,pixel_coords_norm + vec2(pixel_offset.x, 0.0)) :
            ((depth_l > 0.0f) && (depth_l < 1.0f)) ? depthToWorldPos(depth_l,pixel_coords_norm - vec2(pixel_offset.x, 0.0)) : position_c;

        vec3 tangent = normalize(position_t - position_c);
        vec3 bitangent = normalize(position_bt - position_c);

        vec3 normal = normalize(cross(tangent,bitangent));

        imageStore(tgt_tx2D, pixel_coords , vec4(normal,1.0) );
    }
    else{
        imageStore(tgt_tx2D, pixel_coords , vec4(0.0) );
    }
}
