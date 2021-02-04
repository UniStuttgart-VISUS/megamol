#pragma once

//#include "owl/common/math/vec.h"

#include "glm/glm.hpp"

#undef near

namespace megamol {
namespace hpg {
    namespace optix {
        namespace device {
            struct FrameState {
                glm::vec3 camera_screen_du;
                glm::vec3 camera_screen_dv;
                glm::vec3 camera_screen_00;
                glm::vec3 camera_lens_center;

                float th;
                float rw;
                float near;

                int samplesPerPixel{1};
                int maxBounces{0};
                bool accumulate;

                glm::vec4 background;

                int frameIdx;
            };
        } // namespace device
    }     // namespace optix
} // namespace hpg
} // namespace megamol
