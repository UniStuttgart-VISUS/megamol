#include "optix/utils_device.h"

#include "miss.h"
#include "perraydata.h"

namespace megamol {
namespace optix_hpg {
    namespace device {
        MM_OPTIX_MISS_KERNEL(miss_program_occlusion)() {}
    } // namespace device
} // namespace optix_hpg
} // namespace megamol
