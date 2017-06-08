/*
 * IColAdd.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmstd_datatools/AbstractParticleManipulator.h"
#include <vector>
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

    class IColAdd : public stdplugin::datatools::AbstractParticleManipulator {
    public:
        static const char *ClassName(void) { return "IColAdd"; }
        static const char *Description(void) { return "Adds two ICol value streams:  c[] = a_s * a[] + b_s * b[]"; }
        static bool IsAvailable(void) { return true; }

        IColAdd();
        virtual ~IColAdd();

    protected:

        virtual bool manipulateData(
            megamol::core::moldyn::MultiParticleDataCall& outData,
            megamol::core::moldyn::MultiParticleDataCall& inData);

    private:

        core::CallerSlot inDataBSlot;
//        core::param::ParamSlot aOffsetSlot;
        core::param::ParamSlot aScaleSlot;
//        core::param::ParamSlot bOffsetSlot;
        core::param::ParamSlot bScaleSlot;

        size_t inAHash;
        size_t inBHash;
        size_t outHash;
        unsigned int frameID;
        std::vector<float> colors;
        float minCol, maxCol;

    };

}
}
}
