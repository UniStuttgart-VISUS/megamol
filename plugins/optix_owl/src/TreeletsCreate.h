// code originally from https://github.com/UniStuttgart-VISUS/rtxpkd_ldav2020
// modified for MegaMol

// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

// ======================================================================== //
// Modified 2019-2025 VISUS - University of Stuttgart                       //
// ======================================================================== //

#pragma once

#include <functional>
#include <vector>

#include <owl/common/math/box.h>
#include <owl/common/math/vec.h>
#include <owl/common/parallel/parallel_for.h>

#include <tbb/parallel_for.h>

#include "particle.h"

namespace megamol {
namespace optix_owl {
using namespace megamol::optix_owl::device;
using namespace owl::common;

inline std::size_t sort_partition(
    std::vector<device::Particle>& particles, std::size_t begin, std::size_t end, box3f bounds, int& splitDim) {
    // -------------------------------------------------------
    // determine split pos
    // -------------------------------------------------------
    splitDim = arg_max(bounds.span());
    float splitPos = bounds.center()[splitDim];

    // -------------------------------------------------------
    // now partition ...
    // -------------------------------------------------------
    std::size_t mid = begin;
    std::size_t l = begin, r = (end - 1);
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


/*! todo: make this a cmd-line parameter, so we can run scripts to
  measure perf impact per size (already made it a static, so we
  can set it from main() before class is created */
//int TreeletParticles::maxTreeletSize = 1000;

template<typename MakeLeafLambda>
void partitionRecursively(
    std::vector<device::Particle>& particles, std::size_t begin, std::size_t end, const MakeLeafLambda& makeLeaf) {
    if (makeLeaf(begin, end, false))
        // could make into a leaf, done.
        return;

    // -------------------------------------------------------
    // parallel bounding box computation
    // -------------------------------------------------------
    box3f bounds;
    std::mutex boundsMutex;
    parallel_for_blocked(begin, end, 32 * 1024, [&](size_t blockBegin, size_t blockEnd) {
        box3f blockBounds;
        for (size_t i = blockBegin; i < blockEnd; i++)
            blockBounds.extend(particles[i].pos);
        std::lock_guard<std::mutex> lock(boundsMutex);
        bounds.extend(blockBounds);
    });

    int splitDim;
    auto mid = sort_partition(particles, begin, end, bounds, splitDim);

    // -------------------------------------------------------
    // and recurse ...
    // -------------------------------------------------------
    tbb::parallel_for(0, 2, [&](int side) {
        if (side)
            partitionRecursively(particles, begin, mid, makeLeaf);
        else
            partitionRecursively(particles, mid, end, makeLeaf);
    });
}

inline std::vector<PKDlet> prePartition_inPlace(std::vector<device::Particle>& particles, std::size_t maxSize,
    float radius, std::function<bool(box3f const&)> add_cond = nullptr) {
    std::mutex resultMutex;
    std::vector<PKDlet> result;

    if (add_cond == nullptr) {
        partitionRecursively(particles, 0ULL, particles.size(), [&](std::size_t begin, std::size_t end, bool force) {
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
        partitionRecursively(particles, 0ULL, particles.size(), [&](std::size_t begin, std::size_t end, bool force) {
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
} // namespace optix_owl
} // namespace megamol
