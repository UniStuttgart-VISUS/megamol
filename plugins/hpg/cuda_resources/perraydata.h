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
            };
        } // namespace device
    }     // namespace optix
} // namespace hpg
} // namespace megamol
