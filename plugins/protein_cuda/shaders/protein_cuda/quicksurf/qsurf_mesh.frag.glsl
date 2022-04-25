#version 460

#include "protein_gl/deferred/gbuffer_output.glsl"

in vec3 normal;
in vec4 color;

void main() {
    albedo_out = color;
    normal_out = normalize(normal);
    depth_out = gl_FragCoord.z;
    gl_FragDepth = depth_out;
}
