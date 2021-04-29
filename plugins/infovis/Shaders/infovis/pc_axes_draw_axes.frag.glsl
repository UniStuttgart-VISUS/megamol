#version 430

#include "pc_common/pc_extensions.inc.glsl"
#include "pc_common/pc_earlyFragmentTests.inc.glsl"
#include "pc_common/pc_buffers.inc.glsl"
#include "pc_common/pc_uniforms.inc.glsl"

out vec4 fragColor;
in vec4 actualColor;

void main()
{
    fragColor = actualColor;
}
