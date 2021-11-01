#version 440

#include "core/tflookup.inc.glsl"
#include "core/tfconvenience.inc.glsl"
#include "pc_common/pc_extensions.inc.glsl"
#include "pc_common/pc_buffers.inc.glsl"
#include "pc_common/pc_uniforms.inc.glsl"
#include "pc_common/pc_item_draw_histogram_uniforms.inc.glsl"

smooth in vec2 texCoord;
layout(early_fragment_tests) in;
layout(location = 0) out vec4 fragColor;

void main()
{
    #if 0
    vec4 value = clearColor;

    for (uint dimension = 0; dimension < dimensions; ++dimension)
    {
        float axis = scaling.x * abscissae[dimension];
    }
        #endif

    vec4 frags = texture(fragCount, texCoord) - clearColor;

    if (frags.r >= minFragmentCount)
    {
        float value = (frags.r - minFragmentCount) / (maxFragmentCount - minFragmentCount);
        value = clamp(value, 0.0, 1.0);
        fragColor = tflookup(value);
        //fragColor = vec4(vec3(gl_FragCoord.z), 1.0);
    }
    else
    {
        fragColor = clearColor;
    }
}
