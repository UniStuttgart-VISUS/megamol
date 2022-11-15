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
        vec2 from = vec2(0.0, left);
        vec2 to = vec2(1.0, right);
        vec2 lineDir = normalize(to - from);
        vec2 orthLine = normalize(vec2(-lineDir.y, lineDir.x));
        float m = acos(dot(orthLine, vec2(1.0, 0.0)));
        float b = left;
    if(!bitflag_test(flags[itemID], itemTestMask, itemPassMask)){
        // map m and b to image coords
        m = (m - quarterPI) * (thetas) / (halfPI);
        b = b * (rhos-1.0);

        // select bin
        float m_binned = round(m);
        float b_binned = round(b);

        //TODO offset by half texel missing?

        // increment bin density by one
        //uint density = imageAtomicAdd(o_dual_density_tex, ivec3(int(m_binned), int(b_binned), dimID), 1);

        float wt = fract(b);
        float wb = 1.0 - fract(b);
        float wl = 1.0 - fract(m);
        float wr = fract(m);

        imageAtomicAdd(o_dual_density_tex, ivec3(int(floor(m)) + 1, int(floor(b)) , dimID), int(1000.0 * wb * wr) );
        imageAtomicAdd(o_dual_density_tex, ivec3(int(floor(m)) , int(floor(b)) , dimID), int(1000.0 * wb * wl) );
        imageAtomicAdd(o_dual_density_tex, ivec3(int(floor(m)) + 1, int(floor(b)) + 1 , dimID), int(1000.0 * wt * wr) );
        imageAtomicAdd(o_dual_density_tex, ivec3(int(floor(m)), int(floor(b)) +1, dimID), int(1000.0 * wt * wl) );

        //  // compute centroid
        //  float n = float(density+1); //TODO can this even work?
        //  
        //  // compute m and b coords relative to bin
        //  float x_n = m - m_binned;
        //  float y_n = b - b_binned;
        //  
        //  float x_centroid_n_minus_one = fromFixPoint( imageAtomicAdd(o_dual_centroidX_tex, ivec3(int(m_binned), int(b_binned), dimID), toFixPoint(x_n/n)) );
        //  imageAtomicAdd(o_dual_centroidX_tex, ivec3(int(m_binned), int(b_binned), dimID), toFixPoint(-x_centroid_n_minus_one/n) );
        //  float x_centroid_n = x_centroid_n_minus_one + (x_n-x_centroid_n_minus_one)/n;
        //  
        //  float y_centroid_n_minus_one = fromFixPoint( imageAtomicAdd(o_dual_centroidY_tex, ivec3(int(m_binned), int(b_binned), dimID), toFixPoint(y_n/n)) );
        //  imageAtomicAdd(o_dual_centroidY_tex, ivec3(int(m_binned), int(b_binned), dimID), toFixPoint(-y_centroid_n_minus_one/n) );
        //  float y_centroid_n = y_centroid_n_minus_one + (y_n-y_centroid_n_minus_one)/n;
        //  
        //  // compute co-moment based on centroid
        //  imageAtomicAdd(o_dual_comoment_tex, ivec3(int(m_binned), int(b_binned), dimID), toFixPoint( (x_n-x_centroid_n)*(y_n-y_centroid_n_minus_one) ) );
        //  // write to texture at (left, right) atomically
    }
    else{
        imageAtomicAdd(o_select_dualtex, ivec3(int((m - quarterPI) * (thetas) / (halfPI)), int(b * (rhos-1)) , dimID), 1);
    }
}
