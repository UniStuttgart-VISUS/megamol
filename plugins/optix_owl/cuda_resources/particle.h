#pragma once

#include <owl/common/math/vec.h>
#include <owl/common/math/box.h>

#include "CUDAUtils.h"

namespace megamol {
namespace optix_owl {
namespace device {
using namespace owl::common;
struct Particle {
    vec3f pos;
    void set_dim(int dim) {
        pos.x = (pos.x & ~3) | (dim & 3);
    }
    CU_CALLABLE int get_dim() const {
        return pos.x & 3;
    }
};

struct PKDlet {
    //! bounding box of all particles (including the radius)
    box3f bounds;
    //! begin/end range in the common particles array
    unsigned int begin, end;
};
} // namespace device
} // namespace optix_owl
} // namespace megamol
