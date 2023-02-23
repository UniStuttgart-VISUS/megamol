/*
 * ParticleNeighborhoodGraph.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include <cstdint>
#include <vector>

namespace megamol {
namespace core::moldyn {
class MultiParticleDataCall;
} // namespace core::moldyn

namespace datatools {

class ParticleNeighborhoodGraph : public core::Module {
public:
    typedef uint32_t index_t;

    static const char* ClassName() {
        return "ParticleNeighborhoodGraph";
    }
    static const char* Description() {
        return "Computes the direct neighborhood graph of a particle data set";
    }
    static bool IsAvailable() {
        return true;
    }

    ParticleNeighborhoodGraph();
    ~ParticleNeighborhoodGraph() override;

protected:
    bool create() override;
    void release() override;

    bool getData(core::Call& c);
    bool getExtent(core::Call& c);

private:
    void calcData(geocalls::MultiParticleDataCall* data);

    core::CalleeSlot outGraphDataSlot;
    core::CallerSlot inParticleDataSlot;
    core::param::ParamSlot radiusSlot;
    core::param::ParamSlot autoRadiusSlot;
    core::param::ParamSlot autoRadiusSamplesSlot;
    core::param::ParamSlot autoRadiusFactorSlot;
    core::param::ParamSlot autoRadiusSampleRndSeedSlot;
    core::param::ParamSlot forceConnectIsolatedSlot;
    core::param::ParamSlot boundaryXCyclicSlot;
    core::param::ParamSlot boundaryYCyclicSlot;
    core::param::ParamSlot boundaryZCyclicSlot;

    unsigned int frameId;
    size_t inDataHash;
    size_t outDataHash;

    std::vector<index_t> edges;
};

} // namespace datatools
} // namespace megamol
