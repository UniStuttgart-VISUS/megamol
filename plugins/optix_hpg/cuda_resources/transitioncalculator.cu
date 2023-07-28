#include <optix.h>

#include "optix/utils_device.h"
#include "transitioncalculator_device.h"

namespace megamol {
namespace optix_hpg {
    namespace device {
        MM_OPTIX_CLOSESTHIT_KERNEL(tc_closesthit)() {
            const int primID = optixGetPrimitiveIndex();
            // PerRayData& prd = getPerRayData<PerRayData>();

            const auto& self = getProgramData<TransitionCalculatorData>();

            auto ray_state = self.ray_state[primID];

            if (optixIsTriangleFrontFaceHit()) {
                ++(self.mesh_inbound_ctr_ptr[primID]);
                if (ray_state == 0) {
                    self.ray_state[primID] = 2;
                }
                if (ray_state == 1) {
                    self.ray_state[primID] = 3;
                }
            } else if (optixIsTriangleBackFaceHit()) {
                ++(self.mesh_outbound_ctr_ptr[primID]);
                if (ray_state == 0) {
                    self.ray_state[primID] = 1;
                }
                if (ray_state == 2) {
                    self.ray_state[primID] = 4;
                }
            }
        }

        MM_OPTIX_RAYGEN_KERNEL(tc_raygen_program)() {
            const auto& self = getProgramData<TransitionCalculatorData>();
            auto const index = optixGetLaunchIndex();
            auto const dim = optixGetLaunchDimensions();

            auto const rayIdx = index.x + index.y * dim.x;

            if (rayIdx >= self.num_rays)
                return;

            auto const ray = ((Ray*) self.ray_buffer)[rayIdx];

            optixTrace(self.world, (const float3&) ray.origin, (const float3&) ray.direction, ray.tmin, ray.tmax, 0,
                (OptixVisibilityMask) -1,
                /*rayFlags     */ OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                /*SBToffset    */ 0,
                /*SBTstride    */ 1,
                /*missSBTIndex */ 0);
        }

        MM_OPTIX_MISS_KERNEL(tc_miss_program)() {}
    } // namespace device
} // namespace optix_hpg
} // namespace megamol
