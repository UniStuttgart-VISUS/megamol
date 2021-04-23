#version 450

#include <snippet name="::pc::extensions" />
#include <snippet name="::pc::useLineStrip" />
#include <snippet name="::pc::earlyFragmentTests" />
#include <snippet name="::pc::buffers" />
#include <snippet name="::pc::uniforms" />
#include <snippet name="::pc::common" />
#include <snippet name="::pc_item_draw::tessuniforms" />
#include <snippet name="::bitflags::main" />

// Input data
struct Interface
{
    uint itemID;
    vec4 color;
};

layout(early_fragment_tests) in;
layout(location = 0) flat in Interface in_;

layout(location = 0) out vec4 fragColor;
layout(early_fragment_tests) in;

void main()
{
    if (bitflag_test(flags[in_.itemID], fragmentTestMask, fragmentPassMask)) {
        //if (true) {
        fragColor = in_.color;
    } else {
        discard;
        //fragColor = vec4(vec3(0.2), 1.0);
    }
    fragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
