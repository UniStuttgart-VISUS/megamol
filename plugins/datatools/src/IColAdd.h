/*
 * IColAdd.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include <vector>

namespace megamol {
namespace datatools {

class IColAdd : public datatools::AbstractParticleManipulator {
public:
    static const char* ClassName() {
        return "IColAdd";
    }
    static const char* Description() {
        return "Adds two ICol value streams:  c[] = a_s * a[] + b_s * b[]";
    }
    static bool IsAvailable() {
        return true;
    }

    IColAdd();
    ~IColAdd() override;

protected:
    bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;

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

} // namespace datatools
} // namespace megamol
