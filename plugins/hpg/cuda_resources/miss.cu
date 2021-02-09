//#include "owl/owl_device.h"
#include "hpg/optix/utils_device.h"

namespace megamol {
namespace hpg {
    namespace optix {
        namespace device {
            MM_OPTIX_MISS_KERNEL(miss_program)() {
                //printf("MISS\n");
            }
        } // namespace device
    }     // namespace optix
} // namespace hpg
} // namespace megamol
