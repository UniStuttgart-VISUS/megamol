/*
 * ParticleRelaxationModule.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "ParticleRelaxationModule.h"

using namespace megamol;
using namespace megamol::stdplugin;


/*
 * datatools::ParticleRelaxationModule::ParticleRelaxationModule
 */
datatools::ParticleRelaxationModule::ParticleRelaxationModule(void)
        : AbstractParticleManipulator("outData", "indata") {
}


/*
 * datatools::ParticleRelaxationModule::~ParticleRelaxationModule
 */
datatools::ParticleRelaxationModule::~ParticleRelaxationModule(void) {
    this->Release();
}


/*
 * datatools::ParticleRelaxationModule::manipulateData
 */
bool datatools::ParticleRelaxationModule::manipulateData(
        megamol::core::moldyn::MultiParticleDataCall& outData,
        megamol::core::moldyn::MultiParticleDataCall& inData) {
    using megamol::core::moldyn::MultiParticleDataCall;

    outData = inData; // also transfers the unlocker to 'outData'
    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData

    return true;
}
