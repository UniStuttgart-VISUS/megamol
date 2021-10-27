layout(local_size_x = 8, local_size_y = 8) in;

uniform sampler2D src;
layout(rgba16f, binding = 0) writeonly uniform image2D tgt;

void main() {
    ivec2 inPos = ivec2(gl_GlobalInvocationID.xy);

    vec4 value = texelFetch(src, inPos, 0);

    imageStore(tgt, inPos, value);
}