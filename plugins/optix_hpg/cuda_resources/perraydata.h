#pragma once

#include <glm/glm.hpp>
#include <optix.h>

namespace megamol {
namespace optix_hpg {
namespace device {
struct PerRayData {
    int depth;

    glm::vec3 radiance;
    float pdf;

    bool countDepth;
    float ray_depth;

    glm::vec3 origin;
    glm::vec3 direction;

    glm::vec3 wo;

    glm::vec3 beta;

    unsigned int seed;
    int done;

    glm::vec3 lpos;
    glm::vec3 ldir;

    bool countEmitted;
    glm::vec3 emitted;

    float intensity;

    OptixTraversableHandle world;
};
} // namespace device
} // namespace optix_hpg
} // namespace megamol
