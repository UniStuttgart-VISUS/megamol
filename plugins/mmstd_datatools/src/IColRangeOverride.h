/*
 * IColRangeOverride.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmstd_datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

    class IColRangeOverride : public stdplugin::datatools::AbstractParticleManipulator {
    public:
        static const char *ClassName(void) { return "IColRangeOverride"; }
        static const char *Description(void) { return "Sets the ICol min and max values"; }
        static bool IsAvailable(void) { return true; }

        IColRangeOverride();
        virtual ~IColRangeOverride();

    protected:

        virtual bool manipulateData(
            megamol::core::moldyn::MultiParticleDataCall& outData,
            megamol::core::moldyn::MultiParticleDataCall& inData);

    private:

        core::param::ParamSlot overrideSlot;
        core::param::ParamSlot inValsSlot;
        core::param::ParamSlot minValSlot;
        core::param::ParamSlot maxValSlot;
        size_t hash;
        unsigned int frameID;
        float minCol, maxCol;
    };

}
}
}
