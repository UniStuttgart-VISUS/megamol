#version 430

#include "commons.inc.glsl"

in vec4 color;
in vec3 normal;
in vec4 world_pos;
in vec4 clip_plane;
flat in int use_clip_plane;

layout(location = 0) out vec4 frag_color;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out float frag_depth;

void main(void) {
    if (use_clip_plane != 0 && clip_halfspace(world_pos.xyz, clip_plane)) {
        discard;
    }

    frag_color = color;
    frag_normal = normal;
    frag_depth = gl_FragCoord.z;
}
