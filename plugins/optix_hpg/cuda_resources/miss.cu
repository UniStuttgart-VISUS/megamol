#include "optix/utils_device.h"

#include "miss.h"
#include "perraydata.h"

namespace megamol {
namespace optix_hpg {
    namespace device {
        MM_OPTIX_MISS_KERNEL(miss_program)() {
            PerRayData& prd = getPerRayData<PerRayData>();
            const auto& self = getProgramData<MissData>();

            prd.radiance = glm::vec3(self.bg);
            prd.done = true;
        }
    } // namespace device
} // namespace optix_hpg
} // namespace megamol
