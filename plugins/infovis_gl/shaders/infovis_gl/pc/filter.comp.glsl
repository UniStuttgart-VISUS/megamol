#version 450

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

#include "common/common.inc.glsl"
#include "common/invocation_index.inc.glsl"
#include "mmstd_gl/flags/bitflags.inc.glsl"

void main() {
    uint itemIdx = globalInvocationIndex();

    if (itemIdx >= itemCount || !bitflag_test(flags[itemIdx], FLAG_ENABLED, FLAG_ENABLED)) {
        return;
    }

    bool filtered = false;
    for (uint d = 0; d < dimensionCount; d++) {
        float value = pc_dataValue(itemIdx, d);
        if (filters[d].min <= filters[d].max) {
            if (value < filters[d].min || value > filters[d].max) {
                filtered = true;
                break;
            }
        } else {
            if (value < filters[d].min && value > filters[d].max) {
                filtered = true;
                break;
            }
        }
    }

    bitflag_set(flags[itemIdx], FLAG_FILTERED, filtered);
}
