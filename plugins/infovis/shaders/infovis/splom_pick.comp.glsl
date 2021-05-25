#version 430

#include "core/bitflags.inc.glsl"
#include "splom_common/splom_plots.inc.glsl"
#include "splom_common/splom_data.inc.glsl"

uniform vec2 mouse;
uniform int numPlots;
uniform int rowStride;
uniform uint itemCount;
uniform float kernelWidth;
uniform float pickRadius;
uniform int selector;
uniform bool reset;

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

void main()
{
    uint itemID = gl_GlobalInvocationID.y * (gl_NumWorkGroups.x * gl_WorkGroupSize.x) + gl_GlobalInvocationID.x;

    if (itemID >= itemCount || !bitflag_isVisible(flags[itemID])) {
        return;
    }

    if (reset) {
        bitflag_set(flags[itemID], FLAG_SELECTED, false);
        return;
    }

    bool picked = false;
    for (int x = 0; x < numPlots; x++) {
        const Plot plot = plots[x];
        const uint rowOffset = itemID * rowStride;
        vec2 vsPosition = valuesToPosition(plot, vec2(values[rowOffset + plot.indexX], values[rowOffset + plot.indexY])).xy;
        if (distance(mouse, vsPosition) < kernelWidth + pickRadius) {
            picked = true;
            break;
        }
    }
    if (picked) {
        bitflag_set(flags[itemID], FLAG_SELECTED, (selector == 1));
    }
}
