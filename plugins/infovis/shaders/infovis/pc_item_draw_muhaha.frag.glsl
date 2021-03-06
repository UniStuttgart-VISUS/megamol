#version 440

#include "pc_common/pc_extensions.inc.glsl"
#include "pc_common/pc_useLineStrip.inc.glsl"
#include "pc_common/pc_earlyFragmentTests.inc.glsl"
#include "pc_common/pc_buffers.inc.glsl"
#include "pc_common/pc_uniforms.inc.glsl"
#include "pc_common/pc_common.inc.glsl"
#include "core/bitflags.inc.glsl"

// Input data
in Interface
{
#include "pc_common/pc_item_draw_interface.inc.glsl"
} in_;

layout(location = 0) out vec4 fragColor;
layout(early_fragment_tests) in;

void main()
{
    #if 0
    if (bitflag_test(flags[in_.itemID], fragmentTestMask, fragmentPassMask)) {
        //if (bitflag_test(flags[in_.itemID], 5, 1)) {
        fragColor = in_.color;
    } else {
        discard;
        //fragColor = vec4(vec3(0.2), 1.0);
    }
    #endif

    //fragColor = in_.color;
    fragColor = vec4(vec3(gl_FragCoord.z), 1.0);
}
