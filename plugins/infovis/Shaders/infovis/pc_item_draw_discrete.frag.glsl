#version 440

#include <snippet name="::pc::extensions" />
#include <snippet name="::pc::useLineStrip" />
#include <snippet name="::pc::earlyFragmentTests" />
#include <snippet name="::pc::buffers" />
#include <snippet name="::pc::uniforms" />
#include <snippet name="::pc::common" />
#include <snippet name="::bitflags::main" />

// Input data
in Interface
{
#include <snippet name="::pc_item_draw::interface" />
} in_;

layout(location = 0) out vec4 fragColor;
layout(early_fragment_tests) in;
//out float gl_FragDepth;

void main()
{
    if (bitflag_test(flags[in_.itemID], fragmentTestMask, fragmentPassMask)) {
        fragColor = in_.color;
    } else {
        discard;
        //fragColor = vec4(vec3(0.2), 1.0);
    }
}
