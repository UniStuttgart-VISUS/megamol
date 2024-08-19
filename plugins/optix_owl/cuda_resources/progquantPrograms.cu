#include <cuda_runtime.h>
#include <optix_device.h>

#include <owl/common/math/box.h>
#include <owl/common/math/vec.h>
#include <owl/owl_device.h>

#include "intersect.h"
#include "perraydata.h"
#include "progquant.h"

namespace megamol {
namespace optix_owl {
namespace device {
using namespace owl::common;

struct ProgQuantStackEntry {
    float t0, t1;
    unsigned int nodeID;
    box3f refBox;
};

OPTIX_INTERSECT_PROGRAM(progquant_intersect)() {
    const int treeletID = optixGetPrimitiveIndex();
    const auto& self = owl::getProgramData<ProgQuantGeomData>();
    const auto& treelet = self.treeletBuffer[treeletID];
    PerRayData& prd = owl::getPRD<PerRayData>();

    auto const ray = owl::Ray(optixGetWorldRayOrigin(), optixGetWorldRayDirection(), optixGetRayTmin(), optixGetRayTmax());

    const int begin = treelet.begin;
    const int size = treelet.end - begin;
    {
        float t0, t1;
        if (!clipToBounds(ray, treelet.bounds, t0, t1))
            return;


        int nodeID = 0;
        float tmp_hit_t = t1;
        int tmp_hit_primID = -1;

        enum { STACK_DEPTH = 12 };

        ProgQuantStackEntry stackBase[STACK_DEPTH];
        ProgQuantStackEntry* stackPtr = stackBase;

        const int dir_sign[3] = {ray.direction.x < 0.f, ray.direction.y < 0.f, ray.direction.z < 0.f};
        const float org[3] = {ray.origin.x, ray.origin.y, ray.origin.z};
        const float rdir[3] = {
            (fabsf(ray.direction.x) <= 1e-8f) ? 1e8f : 1.f / ray.direction.x,
            (fabsf(ray.direction.y) <= 1e-8f) ? 1e8f : 1.f / ray.direction.y,
            (fabsf(ray.direction.z) <= 1e-8f) ? 1e8f : 1.f / ray.direction.z,
        };

        vec3f pos;
        int dim;

        box3f refBox = treelet.bounds;

        vec3f tmp_hit_pos;

        float compensation = 0.f;

        while (1) {
            // while we have anything to traverse ...

            while (1) {
                // while we can go down

                const int particleID = nodeID + begin;

                {
                    auto const& bpart = self.particleBuffer[particleID];
                    dim = bpart.dim;
                    pos = bpart.from(refBox.span(), refBox.lower);
                }

                compensation = t_compensate(refBox.span()[dim]);

                const float t_slab_lo = (pos[dim] - self.particleRadius - org[dim]) * rdir[dim] - compensation;
                const float t_slab_hi = (pos[dim] + self.particleRadius - org[dim]) * rdir[dim] + compensation;

                const float t_slab_nr = fminf(t_slab_lo, t_slab_hi);
                const float t_slab_fr = fmaxf(t_slab_lo, t_slab_hi);

                // -------------------------------------------------------
                // compute potential sphere interval, and intersect if necessary
                // -------------------------------------------------------
                const float sphere_t0 = fmaxf(t0, t_slab_nr);
                const float sphere_t1 = fminf(fminf(t_slab_fr, t1), tmp_hit_t);

                if (sphere_t0 < sphere_t1) {
                    if (intersectSphere(pos, self.particleRadius, ray, tmp_hit_t)) {
                        tmp_hit_primID = particleID;

                        tmp_hit_pos = pos;
                    }
                }

                // -------------------------------------------------------
                // compute near and far side intervals
                // -------------------------------------------------------
                const float nearSide_t0 = t0;
                const float nearSide_t1 = fminf(fminf(t_slab_fr, t1), tmp_hit_t - compensation);

                const float farSide_t0 = fmaxf(t0, t_slab_nr);
                const float farSide_t1 = fminf(t1, tmp_hit_t + compensation);

                // -------------------------------------------------------
                // logic
                // -------------------------------------------------------
                const int nearSide_nodeID = 2 * nodeID + 1 + dir_sign[dim];
                const int farSide_nodeID = 2 * nodeID + 2 - dir_sign[dim];

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

                    if (dir_sign[dim]) {
                        stackPtr->refBox = leftBounds(refBox, pos[dim], self.particleRadius, dim, compensation);
                        refBox = rightBounds(refBox, pos[dim], self.particleRadius, dim, compensation);
                    } else {
                        stackPtr->refBox = rightBounds(refBox, pos[dim], self.particleRadius, dim, compensation);
                        refBox = leftBounds(refBox, pos[dim], self.particleRadius, dim, compensation);
                    }

                    ++stackPtr;

                    nodeID = nearSide_nodeID;
                    t0 = nearSide_t0;
                    t1 = nearSide_t1;

                    continue;
                }

                nodeID = need_nearSide ? nearSide_nodeID : farSide_nodeID;
                t0 = need_nearSide ? nearSide_t0 : farSide_t0;
                t1 = need_nearSide ? nearSide_t1 : farSide_t1;
                if (dir_sign[dim]) {
                    refBox = need_nearSide ? rightBounds(refBox, pos[dim], self.particleRadius, dim, compensation)
                                           : leftBounds(refBox, pos[dim], self.particleRadius, dim, compensation);
                } else {
                    refBox = need_nearSide ? leftBounds(refBox, pos[dim], self.particleRadius, dim, compensation)
                                           : rightBounds(refBox, pos[dim], self.particleRadius, dim, compensation);
                }
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
                t1 = fminf(t1, tmp_hit_t);

                refBox = stackPtr->refBox;

                if (t1 <= t0)
                    continue;
                break;
            }
        }
    }
}

OPTIX_CLOSEST_HIT_PROGRAM(progquant_ch)() {
    PerRayData& prd = owl::getPRD<PerRayData>();
    prd.particleID = optixGetAttribute_0();
    prd.t = optixGetRayTmax();
}

OPTIX_BOUNDS_PROGRAM(progquant_bounds)(const void* geomData, box3f& primBounds, const int primID) {
    primBounds = ((const ProgQuantGeomData*) geomData)->treeletBuffer[primID].bounds;
}
} // namespace device
} // namespace optix_owl
} // namespace megamol
