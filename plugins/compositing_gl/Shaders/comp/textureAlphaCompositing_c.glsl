uniform sampler2D src0_tx2D; // primary source texture, target texture resolution will match this one
uniform sampler2D src1_tx2D; // secondary source texture, is read from using normalized texture coords derived from primary

uniform float src0_opacity;
uniform float src1_opacity;

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

    vec4 src0_rgba = texture(src0_tx2D,pixel_coords_norm); 
    vec4 src1_rgba = texture(src1_tx2D,pixel_coords_norm);

    vec4 retval;
    retval.a = src0_opacity * src0_rgba.a + (1.0-src0_rgba.a)*src1_rgba.a;
    retval.rgb = src0_rgba.rgb * src0_rgba.a * src0_opacity
                        + src1_rgba.rgb * src1_rgba.a * src1_opacity * (1.0 - src0_opacity * src0_rgba.a);

    imageStore(tgt_tx2D, pixel_coords , retval );
}