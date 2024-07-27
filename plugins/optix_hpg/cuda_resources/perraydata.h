#pragma once

#include <glm/glm.hpp>
#include <optix.h>

namespace megamol {
namespace optix_hpg {
namespace device {
struct PerRayData {
#if 0
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
#else
    int particleID;
    float t;
    glm::vec3 pos;
    glm::vec3 albedo;
    bool countDepth;
    float ray_depth;
#endif
};
} // namespace device
} // namespace optix_hpg
} // namespace megamol
