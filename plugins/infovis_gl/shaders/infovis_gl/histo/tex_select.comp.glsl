#version 430

#include "mmstd_gl/flags/bitflags.inc.glsl"
#include "common.inc.glsl"

uniform sampler2D tex;

layout(std430, binding = 5) buffer Flags
{
    coherent uint flags[];
};

uniform uint numRows = 0;
uniform int selectionMode = 0; // 0 = pick with replace, 1 = append, 2 = remove
uniform int selectedComponent = -1;
uniform int selectedBin = -1;

layout(local_size_x = 32, local_size_y = 32) in;

void main() {
    uint rowId = gl_GlobalInvocationID.y * (gl_NumWorkGroups.x * gl_WorkGroupSize.x) + gl_GlobalInvocationID.x;

    if (rowId >= numRows || !bitflag_isVisible(flags[rowId])) {
        return;
    }

    if (selectedComponent < 0 || selectedBin < 0) {
        if (selectionMode == 0) {
            bitflag_set(flags[rowId], FLAG_SELECTED, false);
        }
        return;
    }

    const ivec2 texSize = textureSize(tex, 0);
    float val = texelFetch(tex, ivec2(rowId % texSize.x, rowId / texSize.x), 0)[selectedComponent];
    val = (val - minimums[selectedComponent]) / (maximums[selectedComponent] - minimums[selectedComponent]);
    int bin_idx = clamp(int(val * numBins), 0, int(numBins) - 1);

    bool isSelected = bin_idx == selectedBin;
    if (selectionMode == 0) {
        bitflag_set(flags[rowId], FLAG_SELECTED, isSelected);
    } else if (selectionMode == 1 && isSelected) {
        bitflag_set(flags[rowId], FLAG_SELECTED, true);
    } else if (selectionMode == 2 && isSelected) {
        bitflag_set(flags[rowId], FLAG_SELECTED, false);
    }
}
