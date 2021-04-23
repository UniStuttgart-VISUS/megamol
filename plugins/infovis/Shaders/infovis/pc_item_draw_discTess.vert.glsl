#version 450

#include <snippet name="::pc::extensions" />
#include <snippet name="::pc::buffers" />
#include <snippet name="::pc::uniforms" />
#include <snippet name="::pc::instancingOffset" />
#include <snippet name="::pc::common" />
#include <snippet name="::pc_item_draw::tessuniforms" />
#include <snippet name="::bitflags::main" />

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
