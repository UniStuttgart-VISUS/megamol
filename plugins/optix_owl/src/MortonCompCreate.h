#pragma once

#include <tuple>
#include <unordered_map>
#include <vector>

#include <owl/common/math/vec.h>

#include "TreeletsCreate.h"

#include "mortoncomp.h"
#include "particle.h"
#include "common.h"

namespace megamol::optix_owl {
using namespace owl::common;

using MortonCode = vec3ui;

std::vector<std::pair<uint64_t, uint64_t>> create_morton_codes(
    std::vector<device::Particle> const& data, box3f const& bounds, device::MortonConfig const& config) {
    std::vector<MortonCode> codes(data.size());

    //constexpr uint64_t const factor = 1 << 21;
    //constexpr uint64_t const factor = 1 << 15;
    auto const dfactor = static_cast<double>(config.factor);

    auto const span = vec3d(bounds.span());
    auto const lower = vec3d(bounds.lower);

#pragma omp parallel for
    for (int64_t i = 0; i < data.size(); ++i) {
        auto const pos = (vec3d(data[i].pos) - lower) / span;
        codes[i] = vec3ui(pos * dfactor);
    }

    std::vector<std::pair<uint64_t, uint64_t>> mc(codes.size());
#pragma omp parallel for
    for (int64_t i = 0; i < codes.size(); ++i) {
        mc[i] = std::make_pair(device::morton_encode(codes[i].x, codes[i].y, codes[i].z), i);
    }

    return mc;
}

void sort_morton_codes(std::vector<std::pair<uint64_t, uint64_t>>& codes) {
    std::sort(codes.begin(), codes.end(), [](auto const& lhs, auto const& rhs) { return lhs.first < rhs.first; });
}

std::tuple<std::vector<std::pair<uint64_t, uint64_t>>, std::vector<uint64_t>, std::vector<device::Particle>>
mask_morton_codes(std::vector<std::pair<uint64_t, uint64_t>> const& codes, std::vector<device::Particle> const& data,
    device::MortonConfig const& config) {
    //constexpr uint32_t const mask = 0b11111111111111110000000000;
    //constexpr uint32_t const mask = 0b1111111111;
    //constexpr uint64_t const mask = 0b111111111111111111000000000000000000000000000000000000000000000;
    //constexpr uint64_t const mask = 0b111111111111111000000000000000000000000000000000000000000000000;

    //constexpr uint64_t const mask = 0b111111111111111000000000000000000000000000000;


    std::unordered_map<uint64_t, uint64_t> grid;

    for (size_t i = 0; i < codes.size(); ++i) {
        ++grid[(codes[i].first & config.mask) >> config.prefix_offset];
        //grid[i] = (codes[i] & mask) >> 30;
    }

    //std::sort(grid.begin(), grid.end(), [](auto const& lhs, auto const& rhs) { return lhs.x < rhs.x; });

    //grid.erase(std::unique(grid.begin(), grid.end()), grid.end());

    std::vector<std::pair<uint64_t, uint64_t>> grid_helper(grid.begin(), grid.end());
    std::sort(
        grid_helper.begin(), grid_helper.end(), [](auto const& lhs, auto const& rhs) { return lhs.first < rhs.first; });

    std::vector<std::pair<uint64_t, uint64_t>> cells(grid_helper.size());
    cells[0].second = grid_helper[0].second;
    for (size_t i = 1; i < grid_helper.size(); ++i) {
        grid_helper[i].second += grid_helper[i - 1].second;
        cells[i].first = grid_helper[i - 1].second;
        cells[i].second = grid_helper[i].second;
    }

    grid.clear();
    grid.reserve(grid_helper.size());
    grid.insert(grid_helper.begin(), grid_helper.end());

    std::vector<uint64_t> sorted_codes(codes.size());
    std::vector<device::Particle> sorted_data(data.size());

    for (size_t i = 0; i < codes.size(); ++i) {
        auto const idx = --grid[(codes[i].first & config.mask) >> config.prefix_offset];
        sorted_codes[idx] = codes[i].first;
        sorted_data[idx] = data[codes[i].second];
    }

    return std::make_tuple(cells, sorted_codes, sorted_data);
}

std::vector<device::PKDlet> prePartition_inPlace(std::vector<device::Particle>& particles, size_t begin, size_t end,
    size_t maxSize, float radius, std::function<bool(box3f const&)> add_cond = nullptr) {
    std::mutex resultMutex;
    std::vector<device::PKDlet> result;

    if (add_cond == nullptr) {
        partitionRecursively(particles, begin, end, [&](size_t begin, size_t end, bool force) {
            /*bool makeLeaf() :*/
            const std::size_t size = end - begin;
            if (size > maxSize && !force)
                return false;

            PKDlet treelet;
            treelet.begin = begin;
            treelet.end = end;
            treelet.bounds = box3f();
            for (std::size_t i = begin; i < end; i++) {
                treelet.bounds.extend(particles[i].pos - radius);
                treelet.bounds.extend(particles[i].pos + radius);
            }

            std::lock_guard<std::mutex> lock(resultMutex);
            result.push_back(treelet);
            return true;
        });
    } else {
        partitionRecursively(
            particles, begin, end, [&](size_t begin, size_t end, bool force) {
                /*bool makeLeaf() :*/
                const std::size_t size = end - begin;
                if (size > maxSize && !force)
                    return false;

                PKDlet treelet;
                treelet.begin = begin;
                treelet.end = end;
                treelet.bounds = box3f();
                for (std::size_t i = begin; i < end; i++) {
                    treelet.bounds.extend(particles[i].pos - radius);
                    treelet.bounds.extend(particles[i].pos + radius);
                }
                if (add_cond(treelet.bounds))
                    return false;

                std::lock_guard<std::mutex> lock(resultMutex);
                result.push_back(treelet);
                return true;
            });
    }

    return std::move(result);
}

std::vector<std::pair<uint64_t, uint64_t>> create_morton_codes(std::vector<device::Particle> const& data,
    size_t begin, size_t end, device::box3f const& bounds, device::MortonConfig const& config) {
    auto const datasize = end - begin;
    std::vector<MortonCode> codes(datasize);

    //constexpr uint64_t const factor = 1 << 21;
    //constexpr uint64_t const factor = 1 << 15;
    auto const dfactor = static_cast<double>(config.factor);

    auto const span = vec3d(bounds.span());
    auto const lower = vec3d(bounds.lower);

    for (size_t i = begin; i < end; ++i) {
        auto const pos = (vec3d(data[i].pos) - lower) / span;
        codes[i - begin] = vec3ui(pos * dfactor);
    }

    std::vector<std::pair<uint64_t, uint64_t>> mc(datasize);
    for (size_t i = begin; i < end; ++i) {
        mc[i - begin] =
            std::make_pair(device::morton_encode(codes[i - begin].x, codes[i - begin].y, codes[i - begin].z), i);
    }

    return mc;
}

std::tuple<std::vector<vec3f>, std::vector<vec3f>, std::vector<vec3f>> convert_morton_treelet(
    device::PKDlet const& treelet, std::vector<device::Particle> const& data, device::MortonCompPKDlet& ctreelet,
    std::vector<device::MortonCompParticle>& cparticles, box3f const& global_bounds,
    device::MortonConfig const& config) {
    auto const codes = create_morton_codes(data, treelet.begin, treelet.end, global_bounds, config);

    //constexpr uint64_t const factor = 1 << 21;
    //constexpr uint64_t const mask = 0b111111111111111000000000000000000000000000000000000000000000000;
    //constexpr uint64_t const mask2 = 0b000000000000000111111111111111111111111111111000000000000000000;

    std::unordered_map<uint64_t, uint64_t> grid;

    for (size_t i = 0; i < codes.size(); ++i) {
        ++grid[(codes[i].first & config.mask) >> config.prefix_offset];
    }

    auto const global_prefix = (codes[0].first & config.mask) >> config.prefix_offset;
    ctreelet.prefix = global_prefix;
    ctreelet.begin = treelet.begin;
    ctreelet.end = treelet.end;
    ctreelet.bounds = treelet.bounds;

    auto const span = global_bounds.span();
    auto const lower = global_bounds.lower;

    std::vector<vec3f> recon_data(codes.size());
    for (size_t i = 0; i < codes.size(); ++i) {
        auto const code = codes[i].first >> config.code_offset;
        auto const prefix = (codes[i].first & config.mask) >> config.prefix_offset;
        if (global_prefix != prefix) {
            throw std::runtime_error("unexpected prefix");
        }

        cparticles[i + treelet.begin].code = code;

        /*auto const combined_code = (static_cast<uint64_t>(cparticles[i + treelet.begin].code) << config.code_offset) +
                                   (static_cast<uint64_t>(ctreelet.prefix) << config.offset);

        uint32_t x, y, z;
        device::morton_decode(combined_code, x, y, z);
        glm::vec3 basePos(x / static_cast<double>(config.factor), y / static_cast<double>(config.factor),
            z / static_cast<double>(config.factor));
        basePos *= global_bounds.span();
        basePos += global_bounds.lower;*/

        auto const basePos = cparticles[i + treelet.begin].from(ctreelet.prefix, span, lower);

        recon_data[i] = basePos;
    }

    std::vector<vec3f> diffs(codes.size());
    std::vector<vec3f> pos(codes.size());
    std::vector<vec3f> rec(codes.size());
    for (size_t i = treelet.begin; i < treelet.end; ++i) {
        diffs[i - treelet.begin] = vec3f(abs(vec3d(data[i].pos) - vec3d(recon_data[i - treelet.begin])));
        pos[i - treelet.begin] = data[i].pos;
        rec[i - treelet.begin] = recon_data[i - treelet.begin];
    }

    return std::make_tuple(pos, rec, diffs);
}

void adapt_morton_bbox(std::vector<device::MortonCompParticle> const& cparticles, device::MortonCompPKDlet& treelet,
    device::box3f const& global_bounds, float const radius, device::MortonConfig const& config) {
    device::box3f bounds;

    auto const span = global_bounds.span();
    auto const lower = global_bounds.lower;

    for (unsigned int i = treelet.begin; i < treelet.end; ++i) {
        bounds.extend(cparticles[i].from(treelet.prefix, span, lower));
    }

    bounds.lower -= radius;
    bounds.upper += radius;

    treelet.bounds = bounds;
}

template<class Comp>
inline void trickle(const Comp& worse, size_t P, device::MortonCompParticle* particle, size_t N, int dim,
    device::MortonCompPKDlet const& treelet, device::box3f const& global_bounds, device::MortonConfig const& config) {
    if (P >= N)
        return;

    auto const span = global_bounds.span();
    auto const lower = global_bounds.lower;

    while (1) {
        const size_t L = lChild(P);
        const size_t R = rChild(P);
        const bool lValid = (L < N);
        const bool rValid = (R < N);

        if (!lValid)
            return;
        size_t C = L;
        if (rValid && worse((particle[R].from(treelet.prefix, span, lower))[dim],
                          (particle[L].from(treelet.prefix, span, lower))[dim]))
            C = R;

        if (!worse((particle[C].from(treelet.prefix, span, lower))[dim],
                (particle[P].from(treelet.prefix, span, lower))[dim]))
            return;

        std::swap(particle[C], particle[P]);
        P = C;
    }
}

template<class Comp>
inline void makeHeap(const Comp& comp, size_t P, device::MortonCompParticle* particle, size_t N, int dim,
    device::MortonCompPKDlet const& treelet, device::box3f const& global_bounds, device::MortonConfig const& config) {
    if (P >= N)
        return;
    const size_t L = lChild(P);
    const size_t R = rChild(P);
    makeHeap(comp, L, particle, N, dim, treelet, global_bounds, config);
    makeHeap(comp, R, particle, N, dim, treelet, global_bounds, config);
    trickle(comp, P, particle, N, dim, treelet, global_bounds, config);
}

void recBuild(size_t /* root node */ P, device::MortonCompParticle* particle, size_t N, box3f const& bounds,
    device::MortonCompPKDlet const& treelet, box3f const& global_bounds, device::MortonConfig const& config) {
    if (P >= N)
        return;

    int dim = arg_max(bounds.upper - bounds.lower);

    auto const span = global_bounds.span();
    auto const lower = global_bounds.lower;

    const size_t L = lChild(P);
    const size_t R = rChild(P);
    const bool lValid = (L < N);
    const bool rValid = (R < N);
    makeHeap(std::greater<float>(), L, particle, N, dim, treelet, global_bounds, config);
    makeHeap(std::less<float>(), R, particle, N, dim, treelet, global_bounds, config);

    auto const P_pos = particle[P].from(treelet.prefix, span, lower);

    if (rValid) {
        while (
            particle[L].from(treelet.prefix, span, lower)[dim] > particle[R].from(treelet.prefix, span, lower)[dim]) {
            std::swap(particle[L], particle[R]);
            trickle(std::greater<float>(), L, particle, N, dim, treelet, global_bounds, config);
            trickle(std::less<float>(), R, particle, N, dim, treelet, global_bounds, config);
        }
        if (particle[L].from(treelet.prefix, span, lower)[dim] > P_pos[dim]) {
            std::swap(particle[L], particle[P]);
            particle[L].dim = dim;
        } else if (particle[R].from(treelet.prefix, span, lower)[dim] < P_pos[dim]) {
            std::swap(particle[R], particle[P]);
            particle[R].dim = dim;
        } else
            /* nothing, root fits */;
    } else if (lValid) {
        if (particle[L].from(treelet.prefix, span, lower)[dim] > P_pos[dim]) {
            std::swap(particle[L], particle[P]);
            particle[L].dim = dim;
        }
    }

    device::box3f lBounds = bounds;
    device::box3f rBounds = bounds;
    lBounds.upper[dim] = rBounds.lower[dim] = P_pos[dim];
    particle[P].dim = dim;

    tbb::parallel_for(0, 2, [&](int childID) {
        if (childID) {
            recBuild(L, particle, N, lBounds, treelet, global_bounds, config);
        } else {
            recBuild(R, particle, N, rBounds, treelet, global_bounds, config);
        }
    });
}

void makePKD(std::vector<device::MortonCompParticle>& particles, device::MortonCompPKDlet const& treelet,
    device::box3f const& global_bounds, device::MortonConfig const& config) {
    recBuild(
        /*node:*/ 0, particles.data() + treelet.begin, treelet.end - treelet.begin, treelet.bounds, treelet,
        global_bounds, config);
}
} // namespace megamol::optix_owl
