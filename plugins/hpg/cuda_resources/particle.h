#pragma once

//#include "owl/common/math/vec.h"
#include "glm/glm.hpp"

namespace megamol {
namespace hpg {
    namespace optix {
        namespace device {
            struct Particle {
                glm::vec4 pos;
            };
        } // namespace device
    }     // namespace optix
} // namespace hpg
} // namespace megamol
