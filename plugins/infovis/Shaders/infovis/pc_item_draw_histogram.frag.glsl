#version 440

#include <snippet name="::core_utils::tflookup" />
#include <snippet name="::core_utils::tfconvenience" />
#include <snippet name="::pc::extensions" />
#include <snippet name="::pc::buffers" />
#include <snippet name="::pc::uniforms" />
#include <snippet name="::pc_item_draw::histogram::uniforms" />

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
