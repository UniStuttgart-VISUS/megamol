#pragma once

#include "optix.h"

#include "framestate.h"
#include "perraydata.h"

#include "glm/glm.hpp"

#include "cuda.h"

namespace megamol {
namespace optix_hpg {
    namespace device {
        struct RayGenData {
            OptixTraversableHandle world;
            int rec_depth;
            glm::uvec2 fbSize;
            glm::vec4* colorBufferPtr;
            FrameState* frameStateBuffer;
            CUsurfObject surface;
        };
    } // namespace device
} // namespace optix_hpg
} // namespace megamol
