#pragma once

#include <limits>
#include <map>
#include <string>
#include <vector>

#include "mmcore/moldyn/MultiParticleDataCall.h"

#include "ospcommon/box.h"


namespace megamol {
namespace ospray {

/*! complete input data for a particle model */
struct MMPKDParticleModel {


    MMPKDParticleModel() : radius(0) { }

    ~MMPKDParticleModel() { }

    std::vector<ospcommon::vec4f> positionf; //!< particle position in float + color encoded in 'w'
    std::vector<ospcommon::vec4d> positiond; //!< particle position in double + color encoded in 'w'

    void fill(megamol::core::moldyn::SimpleSphericalParticles const& parts, bool upgradeToDouble);

    double GetCoord(size_t const idx, size_t const dim) { if (doublePrecision) return positiond[idx][dim]; else return static_cast<double>(positionf[idx][dim]); }

    bool IsDoublePrecision() const { return doublePrecision; }

    ospcommon::box3f GetLocalBBox() const { return lbbox; }

    size_t GetNumParticles() const { return numParticles; }

    float radius; //!< radius to use (0 if not specified)

    bool doublePrecision;

    ospcommon::box3f lbbox;

    size_t numParticles;
};

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
     * a : 8;
     */
    HMMPKDColor_t color;
};

using HMMPKDParticle_t = HMMPKDParticle;

struct HMMPKDParticleModel {
    std::vector<HMMPKDParticle_t> particles;
};

} /* end namespace ospray */
} /* end namespace megamol */
