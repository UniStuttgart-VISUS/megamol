/*
* CUDAGenericFunctions.cuh
* Copyright (C) 2009-2018 by MegaMol Team
* Alle Rechte vorbehalten.
*/
#ifndef MMMOLMAPPLG_CUDAGENERICFUNCTIONS_CUH_INCLUDED
#define MMMOLMAPPLG_CUDAGENERICFUNCTIONS_CUH_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "helper_includes/helper_math.h"
#include "CUDAAdditionalTypedefs.cuh"

namespace megamol {
namespace volume_cuda {

    /** 
     * Intersect ray with a box
     * http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
     * 
     * @param r The ray itself.
     * @param boxmin The minimum values for all three box dimensions.
     * @param boxmax The maximum values for all three box dimensions.
     * @param tnear OUT: Pointer to the distance of the nearest intersection point.
     * @param tfar OUT: Pointer to the distance of the furthest intersection point.
     * @return Value greater 0 if a intersection happened, 0 otherwise.
     */
    static __device__ int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar) {
        // compute intersection of ray with all six bbox planes
        float3 invR = make_float3(1.0f) / r.d;
        float3 tbot = invR * (boxmin - r.o);
        float3 ttop = invR * (boxmax - r.o);

        // re-order intersections to find smallest and largest on each axis
        float3 tmin = fminf(ttop, tbot);
        float3 tmax = fmaxf(ttop, tbot);

        // find the largest tmin and the smallest tmax
        float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
        float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

        *tnear = largest_tmin;
        *tfar = smallest_tmax;

        return smallest_tmax > largest_tmin;
    }

}
}

#endif