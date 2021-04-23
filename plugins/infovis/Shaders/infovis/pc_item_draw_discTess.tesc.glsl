#version 450

#include "pc_common/pc_extensions.inc.glsl"
#include "pc_common/pc_buffers.inc.glsl"
#include "pc_common/pc_uniforms.inc.glsl"
#include "pc_common/pc_common.inc.glsl"
#include "pc_common/pc_item_draw_tessuniforms.inc.glsl"
#include "core/bitflags.inc.glsl"

// BEGIN Output data
layout (vertices = 1) out;
//in InterfaceI
//{
//  flat uint baseItemID;
//} in_[];
//patch out InterfaceTCO
//{
//  uint baseItemID;
//  uint generatedLines;
//} out_;
struct InterfaceTCO
{
    uint baseItemID;
    uint generatedLines;
};
layout(location = 0) patch out InterfaceTCO out_;
// END Output data

void main(void) {
    if (gl_InvocationID == 0) {
        uint baseItemID = gl_PrimitiveID * isoLinesPerInvocation;
        uint remaining = max(itemCount - baseItemID, 0);
        out_.baseItemID = baseItemID;
        out_.generatedLines = remaining;
        gl_TessLevelOuter[0] = remaining;
        gl_TessLevelOuter[1] = dimensionCount;
    }
    gl_out[gl_InvocationID].gl_Position = vec4(1.0);
}
