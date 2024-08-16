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
    //return span / 1023.f / 0.5f;
    return 0.0f;
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

inline void convert_blets(uint64_t P, uint64_t N, Particle const* particles, ProgQuantParticle* out_particles,
    float radius, device::box3f bounds) {
    if (P >= N)
        return;

    auto& out_par = out_particles[P];
    out_par.dim = particles[P].get_dim();

    auto const span = bounds.span();

    out_par.to(particles[P].pos, span, bounds.lower);
    /*auto const diff = (particles[P].pos - bounds.lower) / span;
    out_par.x = static_cast<unsigned int>(diff.x * 1024);
    out_par.y = static_cast<unsigned int>(diff.y * 1024);
    out_par.z = static_cast<unsigned int>(diff.z * 1024);*/

    auto const split_pos = (out_par.get(out_par.dim) / 1023.f) * span[out_par.dim] + bounds.lower[out_par.dim];
    //auto const split_pos = particles[P].pos[out_par.dim];

    auto const lbounds = device::leftBounds(bounds, split_pos, radius, out_par.dim);
    auto const rbounds = device::rightBounds(bounds, split_pos, radius, out_par.dim);

    /*device::box3f lbounds = bounds;
    lbounds.upper[out_par.dim] = split_pos;
    lbounds.upper[out_par.dim] += radius + device::t_compensate(span[out_par.dim]);

    device::box3f rbounds = bounds;
    rbounds.lower[out_par.dim] = split_pos;
    rbounds.lower[out_par.dim] -= radius + device::t_compensate(span[out_par.dim]);*/

    convert_blets(lChild(P), N, particles, out_particles, radius, lbounds);
    convert_blets(rChild(P), N, particles, out_particles, radius, rbounds);
}

inline void reconstruct_blets(uint64_t P, uint64_t N, device::Particle const* particles,
    device::ProgQuantParticle* out_particles, float radius, box3f bounds, vec3f* original, vec3f* rec, vec3f* diffs) {
    if (P >= N)
        return;

    auto& out_par = out_particles[P];

    auto const span = bounds.span();

    auto const pos = out_par.from(span, bounds.lower);

    auto const split_pos = pos[out_par.dim];

    auto const lbounds = device::leftBounds(bounds, split_pos, radius, out_par.dim);
    auto const rbounds = device::rightBounds(bounds, split_pos, radius, out_par.dim);

    vec3d org_pos = vec3d(particles[P].pos);
    vec3d new_pos = vec3d(pos);
    auto const diff = new_pos - org_pos;

    original[P] = vec3f(org_pos);
    rec[P] = vec3f(new_pos);
    diffs[P] = vec3f(diff);

    reconstruct_blets(lChild(P), N, particles, out_particles, radius, lbounds, original, rec, diffs);
    reconstruct_blets(rChild(P), N, particles, out_particles, radius, rbounds, original, rec, diffs);
}
} // namespace device
} // namespace optix_owl
} // namespace megamol
