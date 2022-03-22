/*
 * IColInverse.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include <vector>

namespace megamol {
namespace datatools {

class IColInverse : public datatools::AbstractParticleManipulator {
public:
    static const char* ClassName(void) {
        return "IColInverse";
    }
    static const char* Description(void) {
        return "Inverts the ICol value range.";
    }
    static bool IsAvailable(void) {
        return true;
    }

    IColInverse();
    virtual ~IColInverse();

protected:
    virtual bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData);

private:
    size_t dataHash;
    unsigned int frameID;
    std::vector<float> colors;
    float minCol, maxCol;
};

} // namespace datatools
} // namespace megamol
