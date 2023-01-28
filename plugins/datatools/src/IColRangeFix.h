/*
 * IColRangeFix.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "datatools/AbstractParticleManipulator.h"

namespace megamol::datatools {

class IColRangeFix : public datatools::AbstractParticleManipulator {
public:
    static const char* ClassName() {
        return "IColRangeFix";
    }
    static const char* Description() {
        return "Fixes the ICol min and max values by iterating over all particles";
    }
    static bool IsAvailable() {
        return true;
    }

    IColRangeFix();
    ~IColRangeFix() override;

protected:
    bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;

private:
    size_t hash;
    unsigned int frameID;
    float minCol, maxCol;
};

} // namespace megamol::datatools
