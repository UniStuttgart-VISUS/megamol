#version 430

#include "pc_common/pc_extensions.inc.glsl"
#include "pc_common/pc_useLineStrip.inc.glsl"
#include "pc_common/pc_buffers.inc.glsl"
#include "pc_common/pc_uniforms.inc.glsl"
#include "pc_common/pc_common.inc.glsl"
//#include "::pc_item_filter::uniforms"
#include "core/bitflags.inc.glsl"

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

void main()
{
    uint itemID = gl_GlobalInvocationID.y * (gl_NumWorkGroups.x * gl_WorkGroupSize.x) + gl_GlobalInvocationID.x;


    if (itemID < itemCount && bitflag_test(flags[itemID], FLAG_ENABLED, FLAG_ENABLED))  {

        bitflag_set(flags[itemID], FLAG_FILTERED, false);

        for (uint f = 0; f < dimensionCount; ++f)
        {
            uint dataID = pc_item_dataID(itemID, f);
            float value = pc_item_dataValue_unscaled(dataID);

            if (filters[f].lower <= filters[f].upper) {
                if (value < filters[f].lower || value > filters[f].upper) {
                    bitflag_set(flags[itemID], FLAG_FILTERED, true);
                    break;
                }
            } else {
                if (value < filters[f].lower && value > filters[f].upper) {
                    bitflag_set(flags[itemID], FLAG_FILTERED, true);
                    break;
                }
            }
        }
    }
}
