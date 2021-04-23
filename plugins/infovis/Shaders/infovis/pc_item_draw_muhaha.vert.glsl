#version 440

#include <snippet name="::core_utils::tflookup" />
#include <snippet name="::core_utils::tfconvenience" />
#include <snippet name="::pc::extensions" />
#include <snippet name="::pc::useLineStrip" />
#include <snippet name="::pc::buffers" />
#include <snippet name="::pc::uniforms" />
#include <snippet name="::pc::instancingOffset" />
#include <snippet name="::pc::common" />
#include <snippet name="::bitflags::main" />

// BEGIN Output data
out Interface
{
#include <snippet name="::pc_item_draw::interface" />
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
