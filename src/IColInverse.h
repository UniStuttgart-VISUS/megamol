/*
 * IColInverse.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmstd_datatools/AbstractParticleManipulator.h"
#include <vector>

namespace megamol {
namespace stdplugin {
namespace datatools {

    class IColInverse : public stdplugin::datatools::AbstractParticleManipulator {
    public:
        static const char *ClassName(void) { return "IColInverse"; }
        static const char *Description(void) { return "Inverts the ICol value range."; }
        static bool IsAvailable(void) { return true; }

        IColInverse();
        virtual ~IColInverse();

    protected:

        virtual bool manipulateData(
            megamol::core::moldyn::MultiParticleDataCall& outData,
            megamol::core::moldyn::MultiParticleDataCall& inData);

    private:
        size_t dataHash;
        unsigned int frameID;
        std::vector<float> colors;
        float minCol, maxCol;

    };

}
}
}
