#pragma once

#include <owl/common/math/vec.h>

namespace megamol {
namespace optix_owl {
namespace device {
using namespace owl::common;
struct FrameState {
    vec3f camera_screen_dz;
    vec3f camera_screen_du;
    vec3f camera_screen_dv;
    //vec3f camera_screen_00;
    vec3f camera_lens_center;

    float th;
    float rw;
    float near_plane;

    int accumID;
    int samplesPerPixel = 1;
};
} // namespace device
} // namespace optix_owl
} // namespace megamol
