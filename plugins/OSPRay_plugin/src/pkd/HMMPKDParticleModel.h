#pragma once

#include <cstdint>
#include <vector>

#include "mmcore/moldyn/SimpleSphericalParticles.h"
#include <ospcommon/vec.h>
#include <ospcommon/box.h>

using HMMPKDPosition_t = uint32_t;

using HMMPKDColor_t = uint32_t;

struct HMMPKDParticle {
    /*
     * dim : 2;
     * x : 10;
     * y : 10;
     * z : 10;
     */
    HMMPKDPosition_t position;

    /*
     * r : 8;
     * g : 8:
     * b : 8;
     * ghost : 1
     * a : 7;
     */
    HMMPKDColor_t color;
};

using HMMPKDParticle_t = HMMPKDParticle;

struct HMMPKDParticleModel {
    void fill(megamol::core::moldyn::SimpleSphericalParticles const& parts);

    double GetCoord(size_t const idx, size_t const dim) {
        return position[idx][dim];
    }

    ospcommon::box3f GetLocalBBox() const { return lbbox; }

    size_t GetNumParticles() const { return numParticles; }

    ospcommon::box3f lbbox;

    std::vector<ospcommon::vec4d> position;

    std::vector<HMMPKDParticle_t> pkd_particle;

    float radius;

    size_t numParticles;
};