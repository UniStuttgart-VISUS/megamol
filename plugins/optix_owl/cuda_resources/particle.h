#pragma once

#include <owl/common/math/vec.h>
#include <owl/common/math/box.h>

namespace megamol {
namespace optix_owl {
namespace device {
using namespace owl::common;
struct Particle {
    vec3f pos;
    float dim;
};

struct PKDlet {
    //! bounding box of all particles (including the radius)
    box3f bounds;
    //! begin/end range in the common particles array
    size_t begin, end;
};
} // namespace device
} // namespace optix_owl
} // namespace megamol
