#version 440

#include "core/tflookup.inc.glsl"
#include "core/tfconvenience.inc.glsl"
#include "pc_common/pc_extensions.inc.glsl"
#include "pc_common/pc_useLineStrip.inc.glsl"
#include "pc_common/pc_buffers.inc.glsl"
#include "pc_common/pc_uniforms.inc.glsl"
#include "pc_common/pc_instancingOffset.inc.glsl"
#include "pc_common/pc_common.inc.glsl"
#include "core/bitflags.inc.glsl"

// BEGIN Output data
out Interface
{
#include "pc_common/pc_item_draw_interface.inc.glsl"
} out_;
// END Output data

#include "pc_common/pc_item_draw_vertexShaderMainT.inc.glsl"
