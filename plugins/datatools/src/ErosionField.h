/*
 * ErosionField.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/CallerSlot.h"
#include <vector>

namespace megamol {
namespace datatools {

class ErosionField : public datatools::AbstractParticleManipulator {
public:
    static const char* ClassName() {
        return "ErosionField";
    }
    static const char* Description() {
        return "Computes an erosion "
               "field at the particles based on the neighborhood graph. Points "
               "with ICol < 0.5 are set to zero. The structure of points with "
               "ICol >= 0.5 are eroded and set to the number of iteration the "
               "point was removed.";
    }
    static bool IsAvailable() {
        return true;
    }

    ErosionField();
    ~ErosionField() override;

protected:
    bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inPtData) override;

private:
    core::CallerSlot inNDataSlot;

    size_t inPtHash;
    size_t inNHash;
    size_t outHash;
    unsigned int frameID;
    std::vector<float> colors;
    float maxCol;
};

} // namespace datatools
} // namespace megamol
