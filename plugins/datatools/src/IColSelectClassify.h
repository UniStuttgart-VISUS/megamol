/*
 * IColSelectClassify.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include <vector>

namespace megamol::datatools {

class IColSelectClassify : public datatools::AbstractParticleManipulator {
public:
    static const char* ClassName() {
        return "IColSelectClassify";
    }
    static const char* Description() {
        return "Computes new ICol values: 1 for particles with original ICols close to the selected value; 0 else";
    }
    static bool IsAvailable() {
        return true;
    }

    IColSelectClassify();
    ~IColSelectClassify() override;

protected:
    bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;

private:
    core::param::ParamSlot valueSlot;
    core::param::ParamSlot epsilonSlot;

    size_t inHash;
    size_t outHash;
    unsigned int frameID;
    std::vector<float> colors;
};

} // namespace megamol::datatools
