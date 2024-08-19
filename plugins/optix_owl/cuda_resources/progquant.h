#pragma once

#include <owl/common/math/box.h>
#include <owl/common/math/vec.h>

#include "common.h"
#include "particle.h"

#include "CUDAUtils.h"

namespace megamol {
namespace optix_owl {
namespace device {
using namespace owl::common;
struct ProgQuantParticle {
    unsigned int dim : 2;
    unsigned int x : 10;
    unsigned int y : 10;
    unsigned int z : 10;

    CU_CALLABLE unsigned int get(unsigned int dim) {
        if (dim == 0) {
            return x;
        }
        if (dim == 1) {
            return y;
        }
        if (dim == 2) {
            return z;
        }
    }

    CU_CALLABLE void to(vec3f const& pos, vec3f const& span, vec3f const& lower) {
        auto const diff = (pos - lower) / span;
        x = static_cast<unsigned int>(diff.x * 1023);
        y = static_cast<unsigned int>(diff.y * 1023);
        z = static_cast<unsigned int>(diff.z * 1023);
    }

    CU_CALLABLE vec3f from(vec3f const& span, vec3f const& lower) const {
        vec3f pos(x / 1023.f, y / 1023.f, z / 1023.f);

        return (pos * span) + lower;
    }
};

struct ProgQuantGeomData {
    PKDlet* treeletBuffer;
    ProgQuantParticle* particleBuffer;
    float particleRadius;
};

CU_CALLABLE inline float t_compensate(float span) {
    return span / 1023.f / 0.5f;
    //return 0.0f;
}

CU_CALLABLE inline box3f leftBounds(box3f const& bounds, float split_pos, float radius, int dim) {
    device::box3f lbounds = bounds;
    lbounds.upper[dim] = split_pos;
    lbounds.upper[dim] += radius + t_compensate(bounds.span()[dim]);
    return lbounds;
}

CU_CALLABLE inline box3f rightBounds(box3f const& bounds, float split_pos, float radius, int dim) {
    device::box3f rbounds = bounds;
    rbounds.lower[dim] = split_pos;
    rbounds.lower[dim] -= radius + t_compensate(bounds.span()[dim]);
    return rbounds;
}

CU_CALLABLE inline box3f leftBounds(box3f const& bounds, float split_pos, float radius, int dim, float compensate) {
    device::box3f lbounds = bounds;
    lbounds.upper[dim] = split_pos;
    lbounds.upper[dim] += radius + compensate;
    return lbounds;
}

CU_CALLABLE inline box3f rightBounds(box3f const& bounds, float split_pos, float radius, int dim, float compensate) {
    device::box3f rbounds = bounds;
    rbounds.lower[dim] = split_pos;
    rbounds.lower[dim] -= radius + compensate;
    return rbounds;
}
} // namespace device
} // namespace optix_owl
} // namespace megamol
