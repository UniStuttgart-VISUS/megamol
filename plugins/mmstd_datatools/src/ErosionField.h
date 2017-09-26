/*
 * ErosionField.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmstd_datatools/AbstractParticleManipulator.h"
#include <vector>
#include "mmcore/CallerSlot.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

    class ErosionField : public stdplugin::datatools::AbstractParticleManipulator {
    public:
        static const char *ClassName(void) { return "ErosionField"; }
        static const char *Description(void) { return "Computes an erosion "
            "field at the particles based on the neighborhood graph. Points "
            "with ICol < 0.5 are set to zero. The structure of points with "
            "ICol >= 0.5 are eroded and set to the number of iteration the "
            "point was removed."; }
        static bool IsAvailable(void) { return true; }

        ErosionField();
        virtual ~ErosionField();

    protected:

        virtual bool manipulateData(
            megamol::core::moldyn::MultiParticleDataCall& outData,
            megamol::core::moldyn::MultiParticleDataCall& inPtData);

    private:

        core::CallerSlot inNDataSlot;

        size_t inPtHash;
        size_t inNHash;
        size_t outHash;
        unsigned int frameID;
        std::vector<float> colors;
        float maxCol;

    };

}
}
}
