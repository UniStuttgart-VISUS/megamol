#include "optix/utils_device.h"

#include "particlemeshdistance2_device.h"

#include "optix.h"

namespace megamol {
namespace optix_hpg {
    namespace device {
        MM_OPTIX_CLOSESTHIT_KERNEL(pmd_closesthit)() {
            const int primID = optixGetPrimitiveIndex();
            // PerRayData& prd = getPerRayData<PerRayData>();

            const auto& self = getProgramData<ParticleMeshDistanceData2>();

            auto const index = optixGetLaunchIndex();
            auto const dim = optixGetLaunchDimensions();

            auto const rayIdx = index.x + index.y * dim.x;

            float factor = 1.0f;
            if (optixIsTriangleBackFaceHit())
                factor = -1.0f;

            self.distances[rayIdx] = factor * optixGetRayTmax();
        }

        MM_OPTIX_ANYHIT_KERNEL(pmd_anyhit)
            () {
            const auto& self = getProgramData<ParticleMeshDistanceData2>();

            auto const index = optixGetLaunchIndex();
            auto const dim = optixGetLaunchDimensions();

            auto const rayIdx = index.x + index.y * dim.x;

            /*float factor = 1.0f;
            if (optixIsTriangleBackFaceHit())
                factor = -1.0f;*/

            self.inter_count[rayIdx] = self.inter_count[rayIdx] + 1;
        }

        MM_OPTIX_RAYGEN_KERNEL(pmd_raygen_program)() {
            const auto& self = getProgramData<PMDRayGenData>();
            auto const index = optixGetLaunchIndex();
            auto const dim = optixGetLaunchDimensions();

            auto const rayIdx = index.x + index.y * dim.x;

            if (rayIdx >= self.num_rays)
                return;

            auto const ray = ((Ray*) self.ray_buffer)[rayIdx];

            optixTrace(self.world, (const float3&) ray.origin, (const float3&) ray.direction, ray.tmin, ray.tmax, 0,
                (OptixVisibilityMask) -1,
                /*rayFlags     */ OPTIX_RAY_FLAG_NONE,
                /*SBToffset    */ 0,
                /*SBTstride    */ 1,
                /*missSBTIndex */ 0);
        }

        MM_OPTIX_MISS_KERNEL(pmd_miss_program)() {}
    } // namespace device
} // namespace optix_hpg
} // namespace megamol
