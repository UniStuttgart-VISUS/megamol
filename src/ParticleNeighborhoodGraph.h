/*
 * ParticleNeighborhoodGraph.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/Module.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"
#include <vector>

namespace megamol {
namespace core {
namespace moldyn {
    class MultiParticleDataCall;
}
}

namespace stdplugin {
namespace datatools {

    class ParticleNeighborhoodGraph : public core::Module {
    public:
        static const char *ClassName(void) {
            return "ParticleNeighborhoodGraph";
        }
        static const char *Description(void) {
            return "Computes the direct neighborhood graph of a particle data set";
        }
        static bool IsAvailable(void) {
            return true;
        }

        ParticleNeighborhoodGraph();
        virtual ~ParticleNeighborhoodGraph();

    protected:

        virtual bool create(void);
        virtual void release(void);

        bool getData(core::Call& c);
        bool getExtent(core::Call& c);

    private:

        void calcData(core::moldyn::MultiParticleDataCall* data);

        core::CalleeSlot outGraphDataSlot;
        core::CallerSlot inParticleDataSlot;
        core::param::ParamSlot radiusSlot;
        core::param::ParamSlot autoRadiusSlot;
        core::param::ParamSlot autoRadiusSamplesSlot;
        core::param::ParamSlot autoRadiusFactorSlot;
        core::param::ParamSlot forceConnectIsolatedSlot;

        unsigned int frameId;
        size_t inDataHash;
        size_t outDataHash;

        std::vector<size_t> edges;

    };

}
}
}
