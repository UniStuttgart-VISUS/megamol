#version 430

#include "pc_common/pc_extensions.inc.glsl"
#include "pc_common/pc_useLineStrip.inc.glsl"
#include "pc_common/pc_buffers.inc.glsl"
#include "pc_common/pc_uniforms.inc.glsl"
#include "pc_common/pc_common.inc.glsl"
//#include "::pc_item_stroke::uniforms"
#include "pc_common/pc_item_stroke_intersectLineLine.inc.glsl"
#include "core/bitflags.inc.glsl"

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

void main()
{
    uint itemID = gl_GlobalInvocationID.y * (gl_NumWorkGroups.x * gl_WorkGroupSize.x) + gl_GlobalInvocationID.x;

    if (itemID < itemCount
    && bitflag_isVisible(flags[itemID])
    )
    {
        bool selected = false;

        for (uint dimension = 1; dimension < dimensionCount; ++dimension)
        {
            vec4 p = pc_item_vertex(itemID, pc_item_dataID(itemID, pc_dimension(dimension - 1)), pc_dimension(dimension - 1), (dimension - 1));
            vec4 q = pc_item_vertex(itemID, pc_item_dataID(itemID, pc_dimension(dimension)), pc_dimension(dimension), (dimension));

            if (intersectLineLine(mousePressed, mouseReleased - mousePressed, p.xy, q.xy - p.xy)) {
                selected = true;
                break;
            }
        }

        bitflag_set(flags[itemID], FLAG_SELECTED, selected);
    }
}
