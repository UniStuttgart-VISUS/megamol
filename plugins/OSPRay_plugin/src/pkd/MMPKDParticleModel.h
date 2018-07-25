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

    union {
        std::vector<ospcommon::vec4f> positionf; //!< particle position in float + color encoded in 'w'
        std::vector<ospcommon::vec4d> positiond; //!< particle position in double + color encoded in 'w'
    };

    void fill(megamol::core::moldyn::SimpleSphericalParticles const& parts);

    double GetCoord(size_t const idx, size_t const dim) { if (doublePrecision) return positiond[idx][dim]; else return static_cast<double>(positionf[idx][dim]); }

    bool IsDoublePrecision() const { return doublePrecision; }

    ospcommon::box3f getBounds() const { return bbox; }

    size_t GetNumParticles() const { return numParticles; }

    float radius; //!< radius to use (0 if not specified)

    bool doublePrecision;

    ospcommon::box3f bbox;

    size_t numParticles;
};

} /* end namespace ospray */
} /* end namespace megamol */
