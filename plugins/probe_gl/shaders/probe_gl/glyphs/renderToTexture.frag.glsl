#version 450

#include "probe_gl/glyphs/extensions.inc.glsl"

layout(location = 0) flat in int draw_id;
layout(location = 1) in vec2 uv_coords;
layout(location = 2) in vec3 pixel_vector;
layout(location = 3) in vec3 cam_vector;

layout(location = 0) out vec4 albedo_out;

void main() {
    albedo_out = vec4(0.0,1.0,1.0,1.0);
}
