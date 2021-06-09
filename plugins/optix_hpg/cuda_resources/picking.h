#pragma once

#include "optix.h"

#include "framestate.h"

#include "glm/glm.hpp"

namespace megamol {
namespace optix_hpg {
    namespace device {
        struct PickState {
            glm::uvec2 mouseCoord;
            int primID;
        };

        struct PickingData {
            OptixTraversableHandle world;
            glm::uvec2 fbSize;
            FrameState* frameStateBuffer;
            PickState* pickStateBuffer;
        };
    } // namespace device
} // namespace optix_hpg
} // namespace megamol
