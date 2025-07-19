#pragma once

#include <owl/common/math/box.h>

#include "particle.h"

namespace megamol {
namespace optix_owl {
namespace device {
using namespace owl::common;
struct PKDGeomData {
    Particle* particleBuffer;
    float particleRadius;
    int particleCount;
    box3f worldBounds;
};
} // namespace device
} // namespace optix_owl
} // namespace megamol
