#include "picking.h"
#include "camera.h"

#include "optix/utils_device.h"

namespace megamol {
namespace optix_hpg {
    namespace device {
        MM_OPTIX_RAYGEN_KERNEL(picking_program)() {
            auto& self = getModifiableProgramData<PickingData>();

            const FrameState* fs = &self.frameStateBuffer[0];
            PickState* ps = &self.pickStateBuffer[0];

            float u = -fs->rw + (fs->rw + fs->rw) * float(ps->mouseCoord.x) / self.fbSize.x;
            float v = -(fs->th + (-fs->th - fs->th) * float(ps->mouseCoord.y) / self.fbSize.y);
            auto ray = generateRay(*fs, u, v);

            unsigned int primID = -1;
            optixTrace(self.world, (const float3&) ray.origin, (const float3&) ray.direction, ray.tmin, ray.tmax, 0,
                (OptixVisibilityMask) -1,
                /*rayFlags     */ OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                /*SBToffset    */ 2,
                /*SBTstride    */ MM_OPTIX_SBT_STRIDE,
                /*missSBTIndex */ 1, primID);
            ps->primID = primID;
        }
    } // namespace device
} // namespace optix_hpg
} // namespace megamol
