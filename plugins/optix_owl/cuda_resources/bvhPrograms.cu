#include <cuda_runtime.h>
#include <optix_device.h>

#include <owl/common/math/box.h>
#include <owl/common/math/vec.h>
#include <owl/owl_device.h>

#include "intersect.h"
#include "perraydata.h"
#include "bvh.h"

namespace megamol {
namespace optix_owl {
namespace device {
using namespace owl::common;

OPTIX_INTERSECT_PROGRAM(bvh_intersect)() {
    const int primID = optixGetPrimitiveIndex();
    const auto& self = owl::getProgramData<BVHGeomData>();

    owl::Ray ray(optixGetWorldRayOrigin(), optixGetWorldRayDirection(), optixGetRayTmin(), optixGetRayTmax());

    float tmp_hit_t = ray.tmax;

    const Particle& particle = self.particleBuffer[primID];
    if (intersectSphere(particle, self.particleRadius, ray, tmp_hit_t))
        optixReportIntersection(tmp_hit_t, 0);
}

OPTIX_CLOSEST_HIT_PROGRAM(bvh_ch)() {
    const int primID = optixGetPrimitiveIndex();
    PerRayData& prd = owl::getPRD<PerRayData>();
    const auto& self = owl::getProgramData<BVHGeomData>();
    prd.particleID = primID;
    prd.t = optixGetRayTmax();
    prd.pos = self.particleBuffer[primID].pos;
}

OPTIX_BOUNDS_PROGRAM(bvh_bounds)(const void* geomData, box3f& primBounds, const int primID) {
    auto& self = *(const BVHGeomData*) geomData;
    primBounds.lower = self.particleBuffer[primID].pos - self.particleRadius;
    primBounds.upper = self.particleBuffer[primID].pos + self.particleRadius;
}
} // namespace device
} // namespace optix_owl
} // namespace megamol
