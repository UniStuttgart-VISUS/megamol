#pragma once

#include "particle.h"
#include "perraydata.h"

namespace megamol {
namespace optix_hpg {
namespace device {
struct SphereGeoData {
    Particle* particleBufferPtr;
    glm::vec4* colorBufferPtr;
    float radius;
    bool hasColorData;
    glm::vec4 globalColor;
};
} // namespace device
} // namespace optix_hpg
} // namespace megamol
