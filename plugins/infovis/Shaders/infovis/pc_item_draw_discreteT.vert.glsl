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

#include <snippet name="::pc_item_draw::vertexShaderMainT" />
