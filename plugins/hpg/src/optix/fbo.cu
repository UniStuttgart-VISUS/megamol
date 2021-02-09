#include "cuda.h"

namespace megamol {
namespace hpg {
    namespace optix {
        namespace device {
            inline __device__ void copy_fbo(
                uint32_t* optix_col_buf, float* optix_depth_buf, uint32_t* gl_col_buf, float* gl_depth_buf) {}
        } // namespace device
    }     // namespace optix
} // namespace hpg
} // namespace megamol
