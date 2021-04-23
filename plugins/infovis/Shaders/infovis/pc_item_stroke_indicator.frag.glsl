#version 430

#include "pc_common/pc_extensions.inc.glsl"
#include "pc_common/pc_useLineStrip.inc.glsl"
#include "pc_common/pc_buffers.inc.glsl"
#include "pc_common/pc_uniforms.inc.glsl"
#include "pc_common/pc_common.inc.glsl"
//#include "::pc_item_stroke::uniforms"

uniform vec4 indicatorColor = vec4(0.0, 0.0, 1.0, 1.0);

out vec4 fragColor;

void main()
{
    fragColor = indicatorColor;
}
