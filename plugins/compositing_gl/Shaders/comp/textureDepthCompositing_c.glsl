uniform sampler2D src0_tx2D; // primary source texture, target texture resolution will match this one
uniform sampler2D src1_tx2D; // secondary source texture, is read from using normalized texture coords derived from primary

uniform sampler2D depth0_tx2D;
uniform sampler2D depth1_tx2D;

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

    vec4 rgba0 = texelFetch(src0_tx2D,pixel_coords,0);
    vec4 rgba1 = texelFetch(src1_tx2D,pixel_coords,0);

    float depth0 = texelFetch(depth0_tx2D,pixel_coords,0).r;
    float depth1 = texelFetch(depth1_tx2D,pixel_coords,0).r;

    vec4 front,back,comp;

    if(depth0 > depth1){
        front = rgba1;
        back = rgba0;
    }
    else{
        front = rgba0;
        back = rgba1;
    }

    comp.rgb = ( (front.rgb * front.a) + (back.rgb*back.a*(1.0-front.a)) ) / ( front.a + back.a*(1.0-front.a) );
    comp.a = front.a + back.a*(1.0-front.a);

    imageStore(tgt_tx2D, pixel_coords , comp );
}