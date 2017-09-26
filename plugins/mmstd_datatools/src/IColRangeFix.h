/*
 * IColRangeFix.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmstd_datatools/AbstractParticleManipulator.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

    class IColRangeFix : public stdplugin::datatools::AbstractParticleManipulator {
    public:
        static const char *ClassName(void) { return "IColRangeFix"; }
        static const char *Description(void) { return "Fixes the ICol min and max values by iterating over all particles"; }
        static bool IsAvailable(void) { return true; }

        IColRangeFix();
        virtual ~IColRangeFix();

    protected:

        virtual bool manipulateData(
            megamol::core::moldyn::MultiParticleDataCall& outData,
            megamol::core::moldyn::MultiParticleDataCall& inData);

    private:
        size_t hash;
        unsigned int frameID;
        float minCol, maxCol;
    };

}
}
}
