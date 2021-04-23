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

void main() {
    uint instanceID = gl_VertexID / 6;
    out_.itemID = instanceID;
    float value = out_.itemID / float(itemCount);
    out_.color = mix(color, tflookup(value), tfColorFactor);

    uint local = gl_VertexID % 6;
    uint x = local / 2 - local / 3;
    uint y = local % 2;
    vec4 vertex = vec4(
    margin.x + axisDistance * (dimensionCount - 1) * x,
    margin.y + y * axisHeight,
    pc_item_defaultDepth,
    1.0);

    gl_Position = projection * modelView * vertex;
}
