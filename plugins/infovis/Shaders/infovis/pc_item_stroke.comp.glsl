#version 430

#include <snippet name="::pc::extensions" />
#include <snippet name="::pc::useLineStrip" />
#include <snippet name="::pc::buffers" />
#include <snippet name="::pc::uniforms" />
#include <snippet name="::pc::common" />
#include <snippet name="::pc_item_stroke::uniforms" />
#include <snippet name="::pc_item_stroke::intersectLineLine" />
#include <snippet name="::bitflags::main" />

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
