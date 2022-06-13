#version 450
layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;
#include "common/common.inc.glsl"
#include "common/invocation_index.inc.glsl"

layout(binding = 7, r32ui) uniform coherent uimage2DArray o_dualtex;

void main(){
    uint invID = globalInvocationIndex();
    uint itemID = invID % (2 * itemCount);
    uint dimID = invID / (2* itemCount); // correct division of inv to item/dim
    float left = pc_dataValueNormalized(itemID, dimID);
    float right = pc_dataValueNormalized(itemID, dimID + 1);

    //imageAtomicAdd(o_dualtex, ivec3(left * axisHeight, right* axisHeight, dimID), 1);
    imageAtomicAdd(o_dualtex, ivec3(left * 599, right* 599, dimID), 1);
    // write to texture at (left, right) atomically
}
