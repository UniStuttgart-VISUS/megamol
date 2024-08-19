#pragma once

#include <vector>
#include <unordered_set>
#include <tuple>

#include <owl/common/math/box.h>
#include <owl/common/math/vec.h>

#include "GridCompFixedPoint.h"

#include "particle.h"
#include "gridcomp.h"

#include <omp.h>

namespace megamol::optix_owl {
using namespace owl::common;

struct QPKDParticle {
    unsigned int dim : 2;
    unsigned int sign_x : 1;
    unsigned int sign_y : 1;
    unsigned int sign_z : 1;
    unsigned int x : 16;
    unsigned int y : 16;
    unsigned int z : 16;
};

using decvec3 = owl::common::vec_t<FixedPoint<float, unsigned, dec_val>, 3>;

inline QPKDParticle encode_coord(vec3f const& pos, vec3f const& center, vec3f const& span) {
    decvec3 dec_pos = decvec3(pos);
    decvec3 dec_center = decvec3(center);
    auto const diff = dec_pos;
    //-dec_center;
    QPKDParticle p;
    p.x = diff.x;
    p.y = diff.y;
    p.z = diff.z;
    return p;
}

std::vector<std::pair<size_t, size_t>> gridify(
    std::vector<device::Particle>& data, vec3f const& lower, vec3f const& upper) {
    constexpr float const split_size = (1 << (16 - dec_val)); //    -1.0f;
    auto const span = upper - lower;
    auto const num_cells =
        vec3f(std::ceilf(span.x / split_size), std::ceilf(span.y / split_size), std::ceilf(span.z / split_size));
    auto const diff = span / num_cells;
    std::vector<int> cell_idxs(data.size());
    std::vector<std::vector<size_t>> num_elements(omp_get_max_threads());
    for (auto& el : num_elements) {
        el.resize(num_cells.x * num_cells.y * num_cells.z, 0);
    }
    //std::vector<size_t> num_elements(num_cells.x * num_cells.y * num_cells.z, 0);
    #pragma omp parallel for
    for (int64_t i = 0; i < data.size(); ++i) {
        vec3i cell_idx = vec3i(data[i].pos / diff);
        cell_idx.x = cell_idx.x >= num_cells.x ? num_cells.x - 1 : cell_idx.x;
        cell_idx.y = cell_idx.y >= num_cells.y ? num_cells.y - 1 : cell_idx.y;
        cell_idx.z = cell_idx.z >= num_cells.z ? num_cells.z - 1 : cell_idx.z;
        cell_idxs[i] = cell_idx.x + num_cells.x * (cell_idx.y + num_cells.y * cell_idx.z);
        ++num_elements[omp_get_thread_num()][cell_idxs[i]];
    }
    for (int i = 1; i < num_elements.size(); ++i) {
        std::transform(num_elements[0].begin(), num_elements[0].end(), num_elements[i].begin(), num_elements[0].begin(),
            std::plus());
    }
    std::vector<std::pair<size_t, size_t>> grid_cells(num_elements.size(), std::make_pair(0, 0));
    grid_cells[0].second = num_elements[0][0];
    for (size_t i = 1; i < num_elements[0].size(); ++i) {
        num_elements[0][i] += num_elements[0][i - 1];
        grid_cells[i].first = num_elements[0][i - 1];
        grid_cells[i].second = num_elements[0][i];
    }
    std::vector<device::Particle> tmp(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        auto const idx = cell_idxs[i];
        tmp[--num_elements[0][idx]] = data[i];
    }
    data = tmp;
    /*auto max_x = (*std::max_element(
        cell_idx.begin(), cell_idx.end(), [](auto const& lhs, auto const& rhs) { return lhs.x < rhs.x; })).x;
    auto max_y = (*std::max_element(
        cell_idx.begin(), cell_idx.end(), [](auto const& lhs, auto const& rhs) { return lhs.y < rhs.y; })).y;
    auto max_z = (*std::max_element(
        cell_idx.begin(), cell_idx.end(), [](auto const& lhs, auto const& rhs) { return lhs.z < rhs.z; })).z;*/
    return grid_cells;
}

box3f extendBounds(std::vector<device::Particle> const& particles, size_t begin, size_t end, float radius) {
    box3f bounds;
    for (int64_t p_idx = begin; p_idx < end; ++p_idx) {
        auto const new_lower = particles[p_idx].pos - radius;
        auto const new_upper = particles[p_idx].pos + radius;
        bounds.extend(new_lower);
        bounds.extend(new_upper);
    }

    return bounds;
}

inline size_t spkd_sort_partition(
    std::vector<device::Particle>& particles, size_t begin, size_t end, box3f const& bounds, int& splitDim) {
    // -------------------------------------------------------
    // determine split pos
    // -------------------------------------------------------
    splitDim = arg_max(bounds.upper - bounds.lower);

    float splitPos = (0.5f * (bounds.upper + bounds.lower))[splitDim];

    // -------------------------------------------------------
    // now partition ...
    // -------------------------------------------------------
    size_t mid = begin;
    size_t l = begin, r = (end - 1);
    // quicksort partition:
    while (l <= r) {
        while (l < r && particles[l].pos[splitDim] < splitPos)
            ++l;
        while (l < r && particles[r].pos[splitDim] >= splitPos)
            --r;
        if (l == r) {
            mid = l;
            break;
        }

        std::swap(particles[l], particles[r]);
    }

    // catch-all for extreme cases where all particles are on the same
    // spot, and can't be split:
    if (mid == begin || mid == end)
        mid = (begin + end) / 2;

    return mid;
}

template<typename MakeLeafLambda>
void partition_recurse(
    std::vector<device::Particle>& particles, size_t begin, size_t end, const MakeLeafLambda& makeLeaf) {
    if (makeLeaf(begin, end, false))
        // could make into a leaf, done.
        return;

    // -------------------------------------------------------
    // parallel bounding box computation
    // -------------------------------------------------------
    device::box3f bounds;

    for (size_t idx = begin; idx < end; ++idx) {
        bounds.extend(particles[idx].pos);
    }

    int splitDim;
    auto mid = spkd_sort_partition(particles, begin, end, bounds, splitDim);

    // -------------------------------------------------------
    // and recurse ...
    // -------------------------------------------------------
    tbb::parallel_for(0, 2, [&](int side) {
        if (side)
            partition_recurse(particles, begin, mid, makeLeaf);
        else
            partition_recurse(particles, mid, end, makeLeaf);
    });
}

//inline box3f extendBounds2(
//    std::vector<device::Particle> const& particles, size_t begin, size_t end, float radius) {
//    device::box3f bounds;
//    for (int64_t p_idx = begin; p_idx < end; ++p_idx) {
//        auto const new_lower = particles[p_idx].pos - radius;
//        auto const new_upper = particles[p_idx].pos + radius;
//        bounds.extend(new_lower);
//        bounds.extend(new_upper);
//    }
//
//    return bounds;
//}

inline std::tuple<bool, std::unordered_set<unsigned char>, std::unordered_set<unsigned char>,
    std::unordered_set<unsigned char>>
prefix_consistency(
    std::vector<device::Particle> const& particles, vec3f const& lower, size_t begin, size_t end) {
    std::unordered_set<unsigned char> sx, sy, sz;
    byte_cast bc;
    bc.ui = 0;
    for (size_t i = begin; i < end; ++i) {
        auto const qpkd = encode_coord(particles[i].pos - lower, vec3f(), vec3f());
        bc.ui = qpkd.x;
        sx.insert(bc.parts.b);
        bc.ui = qpkd.y;
        sy.insert(bc.parts.b);
        bc.ui = qpkd.z;
        sz.insert(bc.parts.b);
    }
    return std::make_tuple(sx.size() <= device::spkd_array_size && sy.size() <= device::spkd_array_size &&
                               sz.size() <= device::spkd_array_size,
        sx, sy, sz);
}

inline std::vector<device::GridCompPKDlet> partition_data(std::vector<device::Particle>& particles, size_t tbegin,
    size_t tend, vec3f const& lower, size_t maxSize, float radius) {
    std::mutex resultMutex;
    std::vector<device::GridCompPKDlet> result;

    partition_recurse(particles, tbegin, tend, [&](size_t begin, size_t end, bool force) {
        /*bool makeLeaf() :*/
        const size_t size = end - begin;
        if (size > maxSize && !force)
            return false;

        auto [con, sx, sy, sz] = prefix_consistency(particles, lower, begin, end);
        if (!con)
            return false;

        device::GridCompPKDlet treelet;
        treelet.begin = begin;
        treelet.end = end;
        treelet.bounds = extendBounds(particles, begin, end, radius);
        std::copy(sx.begin(), sx.end(), treelet.sx);
        std::copy(sy.begin(), sy.end(), treelet.sy);
        std::copy(sz.begin(), sz.end(), treelet.sz);

        std::lock_guard<std::mutex> lock(resultMutex);
        result.push_back(treelet);
        return true;
    });

    return std::move(result);
}

template<class Comp>
inline void trickle(
    const Comp& worse, size_t P, device::GridCompParticle* particle, size_t N, int dim, device::GridCompPKDlet const& treelet) {
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
        if (rValid && worse(decode_spart(particle[R], treelet)[dim], decode_spart(particle[L], treelet)[dim]))
            C = R;

        if (!worse(decode_spart(particle[C], treelet)[dim], decode_spart(particle[P], treelet)[dim]))
            return;

        std::swap(particle[C], particle[P]);
        P = C;
    }
}

template<class Comp>
inline void makeHeap(
    const Comp& comp, size_t P, device::GridCompParticle* particle, size_t N, int dim, device::GridCompPKDlet const& treelet) {
    if (P >= N)
        return;
    const size_t L = lChild(P);
    const size_t R = rChild(P);
    makeHeap(comp, L, particle, N, dim, treelet);
    makeHeap(comp, R, particle, N, dim, treelet);
    trickle(comp, P, particle, N, dim, treelet);
}

void recBuild(size_t /* root node */ P, device::GridCompParticle* particle, size_t N, box3f const& bounds,
    device::GridCompPKDlet const& treelet) {
    if (P >= N)
        return;

    int dim = arg_max(bounds.upper - bounds.lower);

    const size_t L = lChild(P);
    const size_t R = rChild(P);
    const bool lValid = (L < N);
    const bool rValid = (R < N);
    makeHeap(std::greater<float>(), L, particle, N, dim, treelet);
    makeHeap(std::less<float>(), R, particle, N, dim, treelet);

    auto const P_pos = decode_spart(particle[P], treelet);

    if (rValid) {
        while (decode_spart(particle[L], treelet)[dim] > decode_spart(particle[R], treelet)[dim]) {
            std::swap(particle[L], particle[R]);
            trickle(std::greater<float>(), L, particle, N, dim, treelet);
            trickle(std::less<float>(), R, particle, N, dim, treelet);
        }
        if (decode_spart(particle[L], treelet)[dim] > P_pos[dim]) {
            std::swap(particle[L], particle[P]);
            particle[L].dim = dim;
        } else if (decode_spart(particle[R], treelet)[dim] < P_pos[dim]) {
            std::swap(particle[R], particle[P]);
            particle[R].dim = dim;
        } else
            /* nothing, root fits */;
    } else if (lValid) {
        if (decode_spart(particle[L], treelet)[dim] > P_pos[dim]) {
            std::swap(particle[L], particle[P]);
            particle[L].dim = dim;
        }
    }

    box3f lBounds = bounds;
    box3f rBounds = bounds;
    lBounds.upper[dim] = rBounds.lower[dim] = P_pos[dim];
    particle[P].dim = dim;

    tbb::parallel_for(0, 2, [&](int childID) {
        if (childID) {
            recBuild(L, particle, N, lBounds, treelet);
        } else {
            recBuild(R, particle, N, rBounds, treelet);
        }
    });
}


void makePKD(std::vector<device::GridCompParticle>& particles, device::GridCompPKDlet const& treelet, size_t begin) {
    recBuild(
        /*node:*/ 0, particles.data() + treelet.begin - begin, treelet.end - treelet.begin, treelet.bounds, treelet);
}

std::tuple<std::vector<vec3f>, std::vector<vec3f>, std::vector<vec3f>> compute_diffs(
    std::vector<device::GridCompPKDlet> const& treelets, std::vector<device::GridCompParticle> const& sparticles,
    std::vector<device::Particle> const& org_data, size_t gbegin, size_t gend) {
    std::vector<vec3f> diffs(gend - gbegin);
    std::vector<vec3f> ops(gend - gbegin);
    std::vector<vec3f> sps(gend - gbegin);
    tbb::parallel_for((size_t) 0, treelets.size(), [&](auto const tID) {
        //for (auto const& treelet : treelets) {
        auto const& treelet = treelets[tID];
        for (size_t i = treelet.begin; i < treelet.end; ++i) {
            vec3d const dpos = vec3d(decode_spart(sparticles[i], treelet));
            vec3d const org_pos = vec3d(org_data[i].pos);
            vec3d const diff = dpos - org_pos;
            diffs[i - gbegin] = vec3f(diff);
            ops[i - gbegin] = vec3f(org_pos);
            sps[i - gbegin] = vec3f(dpos);
        }
        //}
    });
    return std::make_tuple(diffs, ops, sps);
}
} // namespace megamol::optix_owl
