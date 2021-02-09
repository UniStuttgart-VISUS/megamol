#pragma once

#include "framestate.h"

//#include "owl/common/math/vec.h"

#include "hpg/optix/utils_device.h"

namespace megamol {
namespace hpg {
    namespace optix {
        namespace device {

            inline __device__ Ray generateRay(const FrameState& fs, float s, float t) {
                const glm::vec3 origin = fs.camera_lens_center;
                const glm::vec3 direction =
                    (fs.near * fs.camera_screen_00 + s * fs.camera_screen_du + t * fs.camera_screen_dv);

                return Ray(origin, glm::normalize(direction), 1e-6f, 1e20f);
            }
        } // namespace device
    }     // namespace optix
} // namespace hpg
} // namespace megamol
