/*
 * ModColIRange.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "datatools/GraphDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/math/Vector.h"
#include <vector>

namespace megamol {
namespace datatools {

class ModColIRange : public datatools::AbstractParticleManipulator {
public:
    static const char* ClassName(void) {
        return "ModIColRange";
    }
    static const char* Description(void) {
        return "Mapps IColRange values periodically into the specified range.";
    }
    static bool IsAvailable(void) {
        return true;
    }

    ModColIRange();
    ~ModColIRange() override;

protected:
    bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;

private:
    core::param::ParamSlot rangeSlot;

    size_t inDataHash;
    size_t outDataHash;
    unsigned int frameID;
    std::vector<float> colors;
    float minCol, maxCol;
};

} // namespace datatools
} // namespace megamol
