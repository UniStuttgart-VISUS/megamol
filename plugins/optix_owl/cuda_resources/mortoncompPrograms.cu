#include <cuda_runtime.h>
#include <optix_device.h>

#include <owl/common/math/box.h>
#include <owl/common/math/vec.h>
#include <owl/owl_device.h>

#include "intersect.h"
#include "perraydata.h"
#include "mortoncomp.h"

namespace megamol {
namespace optix_owl {
namespace device {
using namespace owl::common;

struct StackEntry {
    float t0, t1;
    unsigned int nodeID;
};

OPTIX_INTERSECT_PROGRAM(mortoncomp_intersect)() {
    const int treeletID = optixGetPrimitiveIndex();
    const auto& self = owl::getProgramData<MortonCompGeomData>();
    const auto treelet = self.treeletBuffer[treeletID];
    PerRayData& prd = owl::getPRD<PerRayData>();

    auto const ray = owl::Ray(optixGetWorldRayOrigin(), optixGetWorldRayDirection(), optixGetRayTmin(), optixGetRayTmax());

    const int begin = treelet.begin;
    const int size = treelet.end - begin;
    {
        float t0, t1;
        if (!clipToBounds(ray, treelet.bounds, t0, t1))
            return;

        int nodeID = 0;
        float tmp_hit_t = ray.tmax;
        int tmp_hit_primID = -1;

        enum { STACK_DEPTH = 12 };

        StackEntry stackBase[STACK_DEPTH];
        StackEntry* stackPtr = stackBase;

        vec3f pos;
        int dim;

        auto const span = self.bounds.span();
        auto const lower = self.bounds.lower;

        //MortonConfig const mc;

        //box3f refBox = treelet.bounds;

        vec3f tmp_hit_pos;

        //float compensation = 0.f;

        while (1) {
            // while we have anything to traverse ...

            while (1) {
                // while we can go down

                const int particleID = nodeID + begin;

                {
                    auto const& cpart = self.particleBuffer[particleID];
                    dim = cpart.dim;
                    pos = cpart.from(treelet.prefix, span, lower);

                    //printf("ISEC %f:%f:%f\n", pos.x, pos.y, pos.z);
                }

                //compensation = t_compensate(refBox.span()[dim]);

                const float t_slab_lo = (pos[dim] - self.particleRadius - ray.origin[dim]) / ray.direction[dim];
                const float t_slab_hi = (pos[dim] + self.particleRadius - ray.origin[dim]) / ray.direction[dim];

                const float t_slab_nr = fminf(t_slab_lo, t_slab_hi);
                const float t_slab_fr = fmaxf(t_slab_lo, t_slab_hi);

                // -------------------------------------------------------
                // compute potential sphere interval, and intersect if necessary
                // -------------------------------------------------------
                /*const float sphere_t0 = fmaxf(t0, t_slab_nr);
                const float sphere_t1 = fminf(fminf(t_slab_fr, t1), tmp_hit_t);*/

                //if (sphere_t0 < sphere_t1) {
                if (intersectSphere(pos, self.particleRadius, ray, tmp_hit_t)) {
                    tmp_hit_primID = particleID;

                    tmp_hit_pos = pos;
                }
                //}

                // -------------------------------------------------------
                // compute near and far side intervals
                // -------------------------------------------------------
                const float nearSide_t0 = t0;
                const float nearSide_t1 = fminf(fminf(t_slab_fr, t1), tmp_hit_t);

                const float farSide_t0 = fmaxf(t0, t_slab_nr);
                const float farSide_t1 = fminf(t1, tmp_hit_t);

                // -------------------------------------------------------
                // logic
                // -------------------------------------------------------
                const int nearSide_nodeID = 2 * nodeID + 1 + (ray.direction[dim] < 0.f);
                const int farSide_nodeID = 2 * nodeID + 2 - (ray.direction[dim] < 0.f);

                const bool nearSide_valid = nearSide_nodeID < size;
                const bool farSide_valid = farSide_nodeID < size;

                const bool need_nearSide = nearSide_valid && nearSide_t0 < nearSide_t1;
                const bool need_farSide = farSide_valid && farSide_t0 < farSide_t1;

                if (!(need_nearSide || need_farSide))
                    break; // pop ...

                if (need_nearSide && need_farSide) {
                    stackPtr->nodeID = farSide_nodeID;
                    stackPtr->t0 = farSide_t0;
                    stackPtr->t1 = farSide_t1;

                    /*if (ray.direction[dim] < 0.f) {
                        stackPtr->refBox = leftBounds(refBox, pos[dim], self.radius, dim, compensation);
                        refBox = rightBounds(refBox, pos[dim], self.radius, dim, compensation);
                    } else {
                        stackPtr->refBox = rightBounds(refBox, pos[dim], self.radius, dim, compensation);
                        refBox = leftBounds(refBox, pos[dim], self.radius, dim, compensation);
                    }*/

                    ++stackPtr;

                    nodeID = nearSide_nodeID;
                    t0 = nearSide_t0;
                    t1 = nearSide_t1;

                    continue;
                }

                nodeID = need_nearSide ? nearSide_nodeID : farSide_nodeID;
                t0 = need_nearSide ? nearSide_t0 : farSide_t0;
                t1 = need_nearSide ? nearSide_t1 : farSide_t1;
                /*if (ray.direction[dim] < 0.f) {
                    refBox = need_nearSide ? rightBounds(refBox, pos[dim], self.radius, dim, compensation)
                                           : leftBounds(refBox, pos[dim], self.radius, dim, compensation);
                } else {
                    refBox = need_nearSide ? leftBounds(refBox, pos[dim], self.radius, dim, compensation)
                                           : rightBounds(refBox, pos[dim], self.radius, dim, compensation);
                }*/
            }
            // -------------------------------------------------------
            // pop
            // -------------------------------------------------------
            while (1) {
                if (stackPtr == stackBase) {
                    // can't pop any more - done.
                    if (tmp_hit_primID >= 0 && tmp_hit_t < ray.tmax) {
                        prd.pos = tmp_hit_pos;
                        optixReportIntersection(tmp_hit_t, 0, tmp_hit_primID);
                    }
                    return;
                }
                --stackPtr;
                //getEntry(stackPtr, nodeID, t0, t1);
                t0 = stackPtr->t0;
                t1 = stackPtr->t1;
                nodeID = stackPtr->nodeID;
                t1 = min(t1, tmp_hit_t);

                //refBox = stackPtr->refBox;

                if (t1 <= t0)
                    continue;
                break;
            }
        }
    }
}

OPTIX_CLOSEST_HIT_PROGRAM(mortoncomp_ch)() {
    PerRayData& prd = owl::getPRD<PerRayData>();
    const auto& self = owl::getProgramData<MortonCompGeomData>();
    prd.particleID = optixGetAttribute_0();
    prd.t = optixGetRayTmax();
}

OPTIX_BOUNDS_PROGRAM(mortoncomp_bounds)(const void* geomData, box3f& primBounds, const int primID) {
    primBounds = ((const MortonCompGeomData*) geomData)->treeletBuffer[primID].bounds;
}
} // namespace device
} // namespace optix_owl
} // namespace megamol
