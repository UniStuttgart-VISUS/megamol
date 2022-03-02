#version 440

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
layout(location = 1) out float selectColor;
layout(early_fragment_tests) in;

void main()
{
    if (bitflag_test(flags[in_.itemID], fragmentTestMask, fragmentPassMask)) {
        fragColor = in_.color;
        // Hack to store selection in a second color attachment, which is maybe faster than using GL_RG32F instead of GL_R32F for the first attachment.
        selectColor = in_.color.g;
    } else {
        discard;
    }
}
