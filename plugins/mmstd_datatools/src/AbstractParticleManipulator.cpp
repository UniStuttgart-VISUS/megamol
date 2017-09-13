/*
 * AbstractParticleManipulator.cpp
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmstd_datatools/AbstractParticleManipulator.h"

using namespace megamol;
using namespace megamol::stdplugin;


/*
 * datatools::AbstractParticleManipulator::AbstractParticleManipulator
 */
datatools::AbstractParticleManipulator::AbstractParticleManipulator(const char *outSlotName, const char *inSlotName) : megamol::core::Module(),
        outDataSlot(outSlotName, "providing access to the manipulated data"),
        inDataSlot(inSlotName, "accessing the original data") {

    this->outDataSlot.SetCallback(megamol::core::moldyn::MultiParticleDataCall::ClassName(), "GetData", &AbstractParticleManipulator::getDataCallback);
    this->outDataSlot.SetCallback(megamol::core::moldyn::MultiParticleDataCall::ClassName(), "GetExtent", &AbstractParticleManipulator::getExtentCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->inDataSlot.SetCompatibleCall<megamol::core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);

}


/*
 * datatools::AbstractParticleManipulator::~AbstractParticleManipulator
 */
datatools::AbstractParticleManipulator::~AbstractParticleManipulator(void) {
}


/*
 * datatools::AbstractParticleManipulator::create
 */
bool datatools::AbstractParticleManipulator::create(void) {
    return true;
}


/*
 * datatools::AbstractParticleManipulator::release
 */
void datatools::AbstractParticleManipulator::release(void) {
}


/*
 * datatools::AbstractParticleManipulator::manipulateData
 */
bool datatools::AbstractParticleManipulator::manipulateData(
        megamol::core::moldyn::MultiParticleDataCall& outData,
        megamol::core::moldyn::MultiParticleDataCall& inData) {
    outData = inData;
    inData.SetUnlocker(nullptr, false);
    return true;
}


/*
 * datatools::AbstractParticleManipulator::manipulateExtent
 */
bool datatools::AbstractParticleManipulator::manipulateExtent(
        megamol::core::moldyn::MultiParticleDataCall& outData,
        megamol::core::moldyn::MultiParticleDataCall& inData) {
    outData = inData;
    inData.SetUnlocker(nullptr, false);
    return true;
}


/*
 * datatools::AbstractParticleManipulator::getDataCallback
 */
bool datatools::AbstractParticleManipulator::getDataCallback(megamol::core::Call& c) {
    using megamol::core::moldyn::MultiParticleDataCall;

    MultiParticleDataCall *outMpdc = dynamic_cast<MultiParticleDataCall*>(&c);
    if (outMpdc == NULL) return false;

    MultiParticleDataCall *inMpdc = this->inDataSlot.CallAs<MultiParticleDataCall>();
    if (inMpdc == NULL) return false;

    *inMpdc = *outMpdc; // to get the correct request time
    if (!(*inMpdc)(0)) return false;

    if (!this->manipulateData(*outMpdc, *inMpdc)) {
        inMpdc->Unlock();
        return false;
    }

    inMpdc->Unlock();

    return true;
}


/*
 * datatools::AbstractParticleManipulator::getExtentCallback
 */
bool datatools::AbstractParticleManipulator::getExtentCallback(megamol::core::Call& c) {
    using megamol::core::moldyn::MultiParticleDataCall;

    MultiParticleDataCall *outMpdc = dynamic_cast<MultiParticleDataCall*>(&c);
    if (outMpdc == NULL) return false;

    MultiParticleDataCall *inMpdc = this->inDataSlot.CallAs<MultiParticleDataCall>();
    if (inMpdc == NULL) return false;

    *inMpdc = *outMpdc; // to get the correct request time
    if (!(*inMpdc)(1)) return false;

    if (!this->manipulateExtent(*outMpdc, *inMpdc)) {
        inMpdc->Unlock();
        return false;
    }

    inMpdc->Unlock();

    return true;
}
