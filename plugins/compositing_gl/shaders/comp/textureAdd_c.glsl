uniform sampler2D src0_tx2D; // primary source texture, target texture resolution will match this one
uniform sampler2D src1_tx2D; // secondary source texture, is read from using normalized texture coords derived from primary

layout(RGBA16) writeonly uniform image2D tgt_tx2D;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    uvec3 gID = gl_GlobalInvocationID.xyz;
    ivec2 pixel_coords = ivec2(gID.xy);
    ivec2 tgt_resolution = imageSize (tgt_tx2D);

    if (pixel_coords.x >= tgt_resolution.x || pixel_coords.y >= tgt_resolution.y) {
        return;
    }

    vec2 pixel_coords_norm = vec2(pixel_coords) / vec2(tgt_resolution);

    vec2 src0_pixel_offset = 0.5 / textureSize(src0_tx2D,0);
    vec2 src1_pixel_offset = 0.5 / textureSize(src1_tx2D,0);

    vec4 retval = texture(src0_tx2D,pixel_coords_norm + src0_pixel_offset) + texture(src1_tx2D,pixel_coords_norm + src1_pixel_offset);

    imageStore(tgt_tx2D, pixel_coords , retval );
}