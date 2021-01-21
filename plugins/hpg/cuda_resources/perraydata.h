#pragma once

//#include "owl/common/math/vec.h"
#include "glm/glm.hpp"

namespace megamol {
namespace hpg {
    namespace optix {
        namespace device {
            struct PerRayData {
                unsigned int primID;
                float t;
                glm::vec3 N;
                glm::vec4 albedo;

                int depth;
                unsigned int seed;

                bool done;
                bool inShadow;

                glm::vec3 radiance;
                glm::vec3 origin;
                glm::vec3 bsdfDir;
                glm::vec3 wo;
                glm::vec3 beta;
                float pdf;
            };
        } // namespace device
    }     // namespace optix
} // namespace hpg
} // namespace megamol
