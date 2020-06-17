layout(std430, binding = 0) buffer FloatData
{
    float floatData[];
};

layout(std430, binding = 1) buffer Minimums
{
    float minimums[];
};

layout(std430, binding = 2) buffer Maximums
{
    float maximums[];
};

layout(std430, binding = 3) buffer Flags
{
    coherent uint flags[];
};

uniform uint binCount = 0;
uniform uint colCount = 0;
uniform uint rowCount = 0;
uniform int selectionMode = 0; // 0 = pick with replace, 1 = append, 2 = remove
uniform int selectedCol = -1;
uniform int selectedBin = -1;

layout(local_size_x = 32, local_size_y = 32) in;

void main()
{
    uint rowId = gl_GlobalInvocationID.y * (gl_NumWorkGroups.x * gl_WorkGroupSize.x) + gl_GlobalInvocationID.x;

    if (rowId >= rowCount || !bitflag_isVisible(flags[rowId])) {
        return;
    }

    if (selectedCol < 0 || selectedBin < 0) {
        if (selectionMode == 0) {
            bitflag_set(flags[rowId], FLAG_SELECTED, false);
        }
        return;
    }

    float val = (floatData[rowId * colCount + selectedCol] - minimums[selectedCol]) / (maximums[selectedCol] - minimums[selectedCol]);
    int bin_idx = clamp(int(val * binCount), 0, int(binCount) - 1);

    bool isSelected = bin_idx == selectedBin;
    if (selectionMode == 0) {
        bitflag_set(flags[rowId], FLAG_SELECTED, isSelected);
    } else if (selectionMode == 1 && isSelected) {
        bitflag_set(flags[rowId], FLAG_SELECTED, true);
    } else if (selectionMode == 2 && isSelected) {
        bitflag_set(flags[rowId], FLAG_SELECTED, false);
    }
}
