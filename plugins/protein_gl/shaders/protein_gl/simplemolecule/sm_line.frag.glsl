#version 430

#include "protein_gl/simplemolecule/sm_common_defines.glsl"
#include "protein_gl/simplemolecule/sm_common_input_frag.glsl"
#include "protein_gl/deferred/gbuffer_output.glsl"

in vec3 move_pos;

void main(void) {
    albedo_out = vec4(move_color.rgb, 1);
    normal_out = vec3(0, 0, 1);
    depth_out = gl_FragCoord.z;
    vec4 ding = vec4(move_pos, 1.0);
    float depth = dot(MVPtransp[2], ding);
    float depthw = dot(MVPtransp[3], ding);
    depth_out = ((depth / depthw) + 1.0) * 0.5;
    gl_FragDepth = depth_out;
}
