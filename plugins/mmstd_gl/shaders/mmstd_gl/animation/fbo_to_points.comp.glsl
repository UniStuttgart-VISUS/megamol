#version 430
#extension GL_ARB_shader_storage_buffer_object : require

#include "point-data.inc.glsl"

layout(binding = 1) uniform sampler2D colTex;
layout(binding = 2) uniform sampler2D depthTex;

uniform mat4 mvp;

layout(local_size_x = 16, local_size_y = 16) in;

void main(void) {
    // which one are we
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    uint outputPos = texelPos.x + textureSize(colTex, 0).x * texelPos.y;

    // grab a sample
    vec4 col = texelFetch(colTex, texelPos, 0);
    float depth = texelFetch(depthTex, texelPos, 0).r;

    // that is the window coordinate, actually with unknown w
    vec2 wincoord = vec2(texelPos);
    vec3 clip_coord = vec3(wincoord, depth);
    clip_coord.xy /= textureSize(colTex, 0).xy;
    clip_coord *= 2.0f;
    clip_coord -= 1.0f;
    mat4 invMVP = inverse(mvp);
    vec4 pos = invMVP * vec4(clip_coord, 1.0f);

    //pos.xyz = clip_coord.xyz;

    // store
    points[outputPos].x = pos.x;
    points[outputPos].y = pos.y;
    points[outputPos].z = pos.z;
    points[outputPos].col = packUnorm4x8(col);
}
