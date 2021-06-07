#version 450

#include "pc_common/pc_extensions.inc.glsl"
#include "pc_common/pc_useLineStrip.inc.glsl"
#include "pc_common/pc_earlyFragmentTests.inc.glsl"
#include "pc_common/pc_buffers.inc.glsl"
#include "pc_common/pc_uniforms.inc.glsl"
#include "pc_common/pc_common.inc.glsl"
#include "pc_common/pc_item_draw_tessuniforms.inc.glsl"
#include "core/bitflags.inc.glsl"

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
