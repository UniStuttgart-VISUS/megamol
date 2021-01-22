#pragma once

//#include "owl/common/math/vec.h"
#include "glm/glm.hpp"

#include "optix.h"

namespace megamol {
namespace hpg {
    namespace optix {
        namespace device {
            struct PerRayData {
                /*unsigned int primID;
                float t;
                glm::vec3 N;
                glm::vec4 albedo;*/

                int depth;

                glm::vec3 result;
                float importance;

                glm::vec3 origin;
                glm::vec3 direction;

                glm::vec3 attenuation;

                unsigned int seed;
                int done;

                glm::vec3 lpos;

                OptixTraversableHandle world;
            };
        } // namespace device
    }     // namespace optix
} // namespace hpg
} // namespace megamol
