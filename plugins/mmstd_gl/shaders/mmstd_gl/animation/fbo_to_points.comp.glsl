#version 430
#extension GL_ARB_shader_storage_buffer_object : require

#include "point-data.inc.glsl"

layout(binding = 1) uniform sampler2D colTex;
layout(binding = 2) uniform sampler2D depthTex;

uniform mat4 mvp;
uniform mat4 projection;
uniform mat4 view;
uniform uint output_offset;

layout(local_size_x = 16, local_size_y = 16) in;

void main(void) {
    // which one are we
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    uint output_pos = texelPos.x + textureSize(colTex, 0).x * texelPos.y;
    output_pos += output_offset;

    // grab a sample
    vec4 col = texelFetch(colTex, texelPos, 0);
    float depth = texelFetch(depthTex, texelPos, 0).r;

    // that is the window coordinate, actually with unknown w
    vec3 clip_coord = vec3(vec2(texelPos), depth);
    clip_coord.xy /= textureSize(colTex, 0).xy;
    clip_coord *= 2.0f;
    clip_coord -= 1.0f;
    vec4 view_pos = inverse(projection) * vec4(clip_coord, 1.0f);
    view_pos /= view_pos.w;
    vec4 pos = inverse(view) * view_pos;

    // store
    points[output_pos].x = pos.x;
    points[output_pos].y = pos.y;
    points[output_pos].z = pos.z;
    points[output_pos].col = packUnorm4x8(col);
}
