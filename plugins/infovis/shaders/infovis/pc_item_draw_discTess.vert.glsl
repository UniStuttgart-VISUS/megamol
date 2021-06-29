#version 450

#include "pc_common/pc_extensions.inc.glsl"
#include "pc_common/pc_buffers.inc.glsl"
#include "pc_common/pc_uniforms.inc.glsl"
#include "pc_common/pc_instancingOffset.inc.glsl"
#include "pc_common/pc_common.inc.glsl"
#include "pc_common/pc_item_draw_tessuniforms.inc.glsl"
#include "core/bitflags.inc.glsl"

// BEGIN Output data
//out Interface
//{
//  flat uint baseItemID;
//} out_;
// END Output data

void main(void) {
    //out_.baseItemID = gl_InstanceID * isoLinesPerInvocation;
    gl_Position = vec4(1);
}
