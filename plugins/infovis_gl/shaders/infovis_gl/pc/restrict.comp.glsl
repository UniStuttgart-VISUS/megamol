#version 450

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

#include "common/common.inc.glsl"
#include "common/invocation_index.inc.glsl"
#include "mmstd_gl/flags/bitflags.inc.glsl"

uniform int restrictionAxis;
uniform float restrictionEpsilon = 0.01;

void main() {
    uint itemIdx = globalInvocationIndex();

    // only enabled and selected items are used to restrict the filter
    if (itemIdx >= itemCount || !bitflag_test(flags[itemIdx], FLAG_ENABLED, FLAG_ENABLED) ||
        !bitflag_test(flags[itemIdx], FLAG_SELECTED, FLAG_SELECTED)) {
        return;
    }

    float valuePlus = pc_dataValueNormalized(itemIdx, restrictionAxis) + restrictionEpsilon;
    float valueMinus = pc_dataValueNormalized(itemIdx, restrictionAxis) - restrictionEpsilon;
    // for some odd reason, using the full range 4294967295 does not work
    uint quantizedValuePlus = uint(valuePlus * double(65535));
    uint quantizedValueMinus = uint(valueMinus * double(65535));
    atomicMin(selectionMin, quantizedValueMinus);
    atomicMax(selectionMax, quantizedValuePlus);
}
