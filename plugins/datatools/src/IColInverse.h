/*
 * IColInverse.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include <vector>

namespace megamol::datatools {

class IColInverse : public datatools::AbstractParticleManipulator {
public:
    static const char* ClassName() {
        return "IColInverse";
    }
    static const char* Description() {
        return "Inverts the ICol value range.";
    }
    static bool IsAvailable() {
        return true;
    }

    IColInverse();
    ~IColInverse() override;

protected:
    bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;

private:
    size_t dataHash;
    unsigned int frameID;
    std::vector<float> colors;
    float minCol, maxCol;
};

} // namespace megamol::datatools
