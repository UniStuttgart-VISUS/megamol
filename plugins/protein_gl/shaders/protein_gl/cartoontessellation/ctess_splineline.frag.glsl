#version 430

#include "protein_gl/deferred/gbuffer_output.glsl"

layout (depth_greater) out float gl_FragDepth; // we think this is right
// this should be wrong //layout (depth_less) out float gl_FragDepth;
#extension GL_ARB_explicit_attrib_location : enable

uniform vec4 lineColor = vec4(1.0, 0.75, 0.2, 1.0);

void main(void) {
    albedo_out = lineColor;
    normal_out = vec3(0.0, 0.0, 0.0);
    depth_out = gl_FragCoord.z;
    gl_FragDepth = depth_out;
}
