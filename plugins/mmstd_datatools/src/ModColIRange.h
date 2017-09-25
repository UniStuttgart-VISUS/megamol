/*
 * ModColIRange.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmstd_datatools/AbstractParticleManipulator.h"
#include <vector>
#include "mmstd_datatools/GraphDataCall.h"
#include "vislib/math/Vector.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

    class ModColIRange : public stdplugin::datatools::AbstractParticleManipulator {
    public:
        static const char *ClassName(void) { return "ModIColRange"; }
        static const char *Description(void) { return "Mapps IColRange values periodically into the specified range."; }
        static bool IsAvailable(void) { return true; }

        ModColIRange();
        virtual ~ModColIRange();

    protected:

        virtual bool manipulateData(
            megamol::core::moldyn::MultiParticleDataCall& outData,
            megamol::core::moldyn::MultiParticleDataCall& inData);

    private:

        core::param::ParamSlot rangeSlot;

        size_t inDataHash;
        size_t outDataHash;
        unsigned int frameID;
        std::vector<float> colors;
        float minCol, maxCol;

    };

}
}
}
