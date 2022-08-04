#version 450

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

#include "common/common.inc.glsl"
#include "common/invocation_index.inc.glsl"

layout(binding = 7, r32ui) uniform coherent uimage2DArray o_dualtex;
uniform int axPxHeight;

void main() {
    uint invID = globalInvocationIndex();
    uint itemID = invID % (itemCount);
    uint dimID = invID / (itemCount); // correct division of inv to item/dim
    float left = pc_dataValueNormalized(itemID, dimID);
    float right = pc_dataValueNormalized(itemID, dimID + 1);
    vec2 from = vec2(0.0, left);
    vec2 to = vec2(1.0, right);
    vec2 lineDir = normalize(to - from);
    vec2 orthLine = normalize(vec2(-lineDir.y, lineDir.x));
    float theta = acos(dot(orthLine, vec2(1.0, 0.0)));
    float rho = abs((from.x * lineDir.y - from.y * lineDir.x)/(orthLine.x * lineDir.y - orthLine.y * lineDir.x));

    //imageAtomicAdd(o_dualtex, ivec3(left * axisHeight, right* axisHeight, dimID), 1);
    imageAtomicAdd(o_dualtex, ivec3(int((theta - 3.141/4.0) * axPxHeight / (3.141/2.0)), int(rho * axPxHeight), dimID), 1);
    // write to texture at (left, right) atomically
}
