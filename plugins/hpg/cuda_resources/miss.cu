//#include "owl/owl_device.h"
#include "hpg/optix/utils_device.h"

#include "miss.h"
#include "perraydata.h"

namespace megamol {
namespace hpg {
    namespace optix {
        namespace device {
            MM_OPTIX_MISS_KERNEL(miss_program)() {
                PerRayData& prd = getPerRayData<PerRayData>();
                const auto& self = getProgramData<MissData>();

                prd.radiance = glm::vec3(self.bg);
                prd.done = true;
            }

            MM_OPTIX_MISS_KERNEL(miss_program_occlusion)() {

            }
        } // namespace device
    }     // namespace optix
} // namespace hpg
} // namespace megamol
