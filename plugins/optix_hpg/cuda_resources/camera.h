#pragma once

#include "framestate.h"

#include "optix/utils_device.h"

namespace megamol {
namespace optix_hpg {
namespace device {
// rtxpkd
inline __device__ Ray generateRay(const FrameState& fs, float s, float t) {
    const glm::vec3 origin = fs.camera_center;
    const glm::vec3 direction = (fs.near * fs.camera_front + s * fs.camera_right + t * fs.camera_up);

    return Ray(origin, glm::normalize(direction), 1e-6f, 1e20f);
}

} // namespace device
} // namespace optix_hpg
} // namespace megamol
