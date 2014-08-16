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
        : AbstractParticleManipulator("outData", "indata"), tfq(),
        dataHash(0), frameId(0) {
    this->MakeSlotAvailable(this->tfq.GetSlot());
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

    if ((this->frameId != inData.FrameID()) || (this->dataHash != inData.DataHash()) || (inData.DataHash() == 0)) {
        this->frameId = inData.FrameID();
        this->dataHash = inData.DataHash();
        // Data updated. Need refresh.

        // TODO: Implement!

        //this->setData(inData);
    }

    //outData = inData; // also transfers the unlocker to 'outData'
    //inData.SetUnlocker(nullptr, false); // keep original data locked
    //                                    // original data will be unlocked through outData

    return true;
}
