#pragma once

#include <owl/common/math/box.h>

#include "particle.h"

namespace megamol {
namespace optix_owl {
namespace device {
using namespace owl::common;
struct BVHGeomData {
    Particle* particleBuffer;
    float particleRadius;
};
} // namespace device
} // namespace optix_owl
} // namespace megamol
