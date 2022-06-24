/*
 * ParticleListSelector.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#include "ParticleListSelector.h"
#include "mmcore/param/IntParam.h"

using namespace megamol;


/*
 * datatools::ParticleListSelector::ParticleListSelector
 */
datatools::ParticleListSelector::ParticleListSelector(void)
        : AbstractParticleManipulator("outData", "indata")
        , listIndexSlot("listIndex", "The thinning factor. Only each n-th particle will be kept.") {
    this->listIndexSlot.SetParameter(new core::param::IntParam(0, 0));
    this->MakeSlotAvailable(&this->listIndexSlot);
}


/*
 * datatools::ParticleListSelector::~ParticleListSelector
 */
datatools::ParticleListSelector::~ParticleListSelector(void) {
    this->Release();
}


/*
 * datatools::ParticleListSelector::manipulateData
 */
bool datatools::ParticleListSelector::manipulateData(
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {
    using geocalls::MultiParticleDataCall;
    int idx = this->listIndexSlot.Param<core::param::IntParam>()->Value();

    outData = inData; // also transfers the unlocker to 'outData'

    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData

    if ((idx < 0) || (idx >= static_cast<int>(inData.GetParticleListCount()))) {
        outData.SetParticleListCount(0);
    } else {
        outData.SetParticleListCount(1);
        outData.AccessParticles(0) = inData.AccessParticles(idx);
    }

    return true;
}
