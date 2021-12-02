/*
 * IColRangeOverride.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace datatools {

class IColRangeOverride : public datatools::AbstractParticleManipulator {
public:
    static const char* ClassName(void) {
        return "IColRangeOverride";
    }
    static const char* Description(void) {
        return "Sets the ICol min and max values";
    }
    static bool IsAvailable(void) {
        return true;
    }

    IColRangeOverride();
    virtual ~IColRangeOverride();

protected:
    virtual bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData);

private:
    core::param::ParamSlot overrideSlot;
    core::param::ParamSlot inValsSlot;
    core::param::ParamSlot minValSlot;
    core::param::ParamSlot maxValSlot;
    size_t hash;
    unsigned int frameID;
    float minCol, maxCol;
};

} // namespace datatools
} // namespace megamol
