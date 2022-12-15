#version 430
#extension GL_ARB_shader_storage_buffer_object : require

struct point {
    float x;
    float y;
    float z;
    // packed unorm 4x8
    uint col;
};

layout(std430, binding = 1) buffer pointData {
    point points[];
};

layout(binding = 1) uniform sampler2D colTex;
layout(binding = 2) uniform sampler2D depthTex;

layout(local_size_x = 16, local_size_y = 16) in;

void main(void) {
    // which one are we
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    uint outputPos = texelPos.x + textureSize(colTex, 0).x * texelPos.y;

    // grab a sample
    vec4 col = texelFetch(colTex, texelPos, 0);
    float depth = texelFetch(depthTex, texelPos, 0).r;

    // TODO: unproject
    vec4 pos = vec4(texelPos, depth, 1.0);

    // store
    points[outputPos].x = pos.x;
    points[outputPos].y = pos.y;
    points[outputPos].z = pos.z;
    points[outputPos].col = packUnorm4x8(col);
}
