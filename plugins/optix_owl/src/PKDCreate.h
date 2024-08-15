#pragma once

#include <vector>

#include <owl/common/math/vec.h>
#include <owl/common/math/box.h>
#include <owl/common/parallel/parallel_for.h>

#include <tbb/parallel_for.h>

#include "particle.h"

namespace megamol {
namespace optix_owl {
using namespace megamol::optix_owl::device;
using namespace owl::common;

inline float computeStableEpsilon(float f) {
    return abs(f) * float(1. / (1 << 21));
}

inline float computeStableEpsilon(const owl::common::vec3f v) {
    return max(max(computeStableEpsilon(v.x), computeStableEpsilon(v.y)), computeStableEpsilon(v.z));
}

inline size_t lChild(size_t P) {
    return 2 * P + 1;
}
inline size_t rChild(size_t P) {
    return 2 * P + 2;
}

template<class Comp>
inline void trickle(const Comp& worse, size_t P, Particle* particle, size_t N, int dim) {
    if (P >= N)
        return;

    while (1) {
        const size_t L = lChild(P);
        const size_t R = rChild(P);
        const bool lValid = (L < N);
        const bool rValid = (R < N);

        if (!lValid)
            return;
        size_t C = L;
        if (rValid && worse(particle[R].pos[dim], particle[L].pos[dim]))
            C = R;

        if (!worse(particle[C].pos[dim], particle[P].pos[dim]))
            return;

        std::swap(particle[C], particle[P]);
        P = C;
    }
}

template<class Comp>
inline void makeHeap(const Comp& comp, size_t P, Particle* particle, size_t N, int dim) {
    if (P >= N)
        return;
    const size_t L = lChild(P);
    const size_t R = rChild(P);
    makeHeap(comp, L, particle, N, dim);
    makeHeap(comp, R, particle, N, dim);
    trickle(comp, P, particle, N, dim);
}

inline void recBuild(size_t /* root node */ P, Particle* particle, size_t N, box3f bounds) {
    if (P >= N)
        return;

    int dim = arg_max(bounds.span());

    const size_t L = lChild(P);
    const size_t R = rChild(P);
    const bool lValid = (L < N);
    const bool rValid = (R < N);
    makeHeap(std::greater<float>(), L, particle, N, dim);
    makeHeap(std::less<float>(), R, particle, N, dim);

    if (rValid) {
        while (particle[L].pos[dim] > particle[R].pos[dim]) {
            std::swap(particle[L], particle[R]);
            trickle(std::greater<float>(), L, particle, N, dim);
            trickle(std::less<float>(), R, particle, N, dim);
        }
        if (particle[L].pos[dim] > particle[P].pos[dim]) {
            std::swap(particle[L], particle[P]);
            particle[L].dim = dim;
        } else if (particle[R].pos[dim] < particle[P].pos[dim]) {
            std::swap(particle[R], particle[P]);
            particle[R].dim = dim;
        } else
            /* nothing, root fits */;
    } else if (lValid) {
        if (particle[L].pos[dim] > particle[P].pos[dim]) {
            std::swap(particle[L], particle[P]);
            particle[L].dim = dim;
        }
    }

    box3f lBounds = bounds;
    box3f rBounds = bounds;
    lBounds.upper[dim] = rBounds.lower[dim] = particle[P].pos[dim];
    particle[P].dim = dim;

    tbb::parallel_for(0, 2, [&](int childID) {
        if (childID) {
            recBuild(L, particle, N, lBounds);
        } else {
            recBuild(R, particle, N, rBounds);
        }
    });

    /*parallel_for(2, [&](int childID) {
        if (childID) {
            recBuild(L, particle, N, lBounds);
        } else {
            recBuild(R, particle, N, rBounds);
        }
    });*/
}

// /*! build pkd for an _entire_ std::vector */

inline void makePKD(std::vector<Particle>& particles, box3f bounds) {
    recBuild(/*node:*/ 0, particles.data(), particles.size(), bounds);
}

/*! make a pkd treelet _inside_ a particle array, for the given
    begin/end range. ie, the particle at position 'begin' will be
    the root node of a tree of (end-ebgin) paricles. Note this is
    _NOT_ the same as a pkd subtree rooted at position begin (for
    the latter case, the children of that node would be at 2*begin+1
    and 2*begin+2, while for the complete tree built by this
    function they'll be at begin+1 and begin+2. */

inline void makePKD(std::vector<Particle>& particles, size_t begin, size_t end, box3f bounds) {
    recBuild(/*node:*/ 0, particles.data() + begin, end - begin, bounds);
}
} // namespace optix_owl
} // namespace megamol
