#version 450

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

#include "common/common.inc.glsl"
#include "common/invocation_index.inc.glsl"
#include "mmstd_gl/flags/bitflags.inc.glsl"

layout(binding = 7, r32ui) uniform coherent uimage2DArray o_dual_density_tex;
layout(binding = 8, r32ui) uniform coherent uimage2DArray o_dual_centroidX_tex;
layout(binding = 9, r32ui) uniform coherent uimage2DArray o_dual_centroidY_tex;
layout(binding = 10, r32ui) uniform coherent uimage2DArray o_dual_comoment_tex;
layout(binding = 6, r32ui) uniform coherent uimage2DArray o_select_dualtex;
uniform int axPxHeight;
uniform int thetas;
uniform int rhos;
uniform uint itemTestMask = 0;
uniform uint itemPassMask = 0;

#define PI        3.1415926538
#define halfPI    1.5707963268
#define quarterPI 0.7853981634

uint toFixPoint(float v){
    return int(v * 1000.0);
}

float fromFixPoint(uint v){
    return float(v) / 1000.0;
}

void main() {
    uint invID = globalInvocationIndex();
    uint itemID = invID % (itemCount);

    uint dimID = invID / (itemCount); // correct division of inv to item/dim
    float left = pc_dataValueNormalized(itemID, dimID);
    float right = pc_dataValueNormalized(itemID, dimID + 1);

    if(!bitflag_test(flags[itemID], itemTestMask, itemPassMask)){
        float left_axis = (left * (thetas-1));
        float right_axis = (right * (rhos-1));

        float wr = fract(left_axis);
        float wl = 1.0 - fract(left_axis);
        float wb = 1.0 - fract(right_axis);
        float wt = fract(right_axis);

        imageAtomicAdd(o_dual_density_tex, ivec3(int(floor(left_axis)) + 1, int(floor(right_axis))    , dimID), toFixPoint(wb * wr) );
        imageAtomicAdd(o_dual_density_tex, ivec3(int(floor(left_axis))    , int(floor(right_axis))    , dimID), toFixPoint(wb * wl) );
        imageAtomicAdd(o_dual_density_tex, ivec3(int(floor(left_axis)) + 1, int(floor(right_axis)) + 1, dimID), toFixPoint(wt * wr) );
        imageAtomicAdd(o_dual_density_tex, ivec3(int(floor(left_axis))    , int(floor(right_axis)) + 1, dimID), toFixPoint(wt * wl) );

    }
    else{
        //imageAtomicAdd(o_select_dualtex, ivec3(int((m - quarterPI) * (thetas) / (halfPI)), int(b * (rhos-1)) , dimID), 1);
    }
}
