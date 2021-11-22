#version 430

#include "protein_gl/simplemolecule/sm_common_defines.glsl"
#include "protein_gl/deferred/gbuffer_output.glsl"

layout (depth_greater) out float gl_FragDepth; // we think this is right
// this should be wrong //layout (depth_less) out float gl_FragDepth;
#extension GL_ARB_explicit_attrib_location : enable

in vec4 mycol;
in vec3 rawnormal;

void main(void) {
    albedo_out = mycol;    
    normal_out = normalize(rawnormal);
    depth_out = gl_FragCoord.z;
    gl_FragDepth = depth_out;
}
