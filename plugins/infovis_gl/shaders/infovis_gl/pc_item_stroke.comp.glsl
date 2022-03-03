#version 430

#include "pc_common/pc_buffers.inc.glsl"
#include "pc_common/pc_uniforms.inc.glsl"
#include "pc_common/pc_common.inc.glsl"
#include "core/bitflags.inc.glsl"

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

// @see http://stackoverflow.com/a/565282/791895
#define FLOAT_EPS (1.0e-10)

float cross2(vec2 v, vec2 w) {
    return v.x * w.y - v.y * w.x;
}

bool intersectLineLine(vec2 p, vec2 r, vec2 q, vec2 s) {
    float rXs = cross2(r, s);

    if (abs(rXs) > FLOAT_EPS) {
        vec2 qp = q - p;
        float t = cross2(qp, s) / rXs;
        float u = cross2(qp, r) / rXs;

        return (0.0 <= t) && (t <= 1.0) && (0.0 <= u) && (u <= 1.0);
    }

    return false;
}

void main() {
    uint itemID = gl_GlobalInvocationID.y * (gl_NumWorkGroups.x * gl_WorkGroupSize.x) + gl_GlobalInvocationID.x;

    if (itemID < itemCount && bitflag_isVisible(flags[itemID])) {
        bool selected = false;

        for (uint dimension = 1; dimension < dimensionCount; ++dimension) {
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
