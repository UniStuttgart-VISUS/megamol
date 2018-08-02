/*
 * AbstractParticleManipulator.cpp
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmstd_datatools/AbstractVolumeManipulator.h"
#include "mmstd_datatools/AbstractParticleManipulator.h"

using namespace megamol;
using namespace megamol::stdplugin;


/*
 * datatools::AbstractVolumeManipulator::AbstractVolumeManipulator
 */
datatools::AbstractVolumeManipulator::AbstractVolumeManipulator(const char* outSlotName, const char* inSlotName)
    : megamol::core::Module()
    ,
        outDataSlot(outSlotName, "providing access to the manipulated data"),
        inDataSlot(inSlotName, "accessing the original data") {

    this->outDataSlot.SetCallback(
        megamol::core::moldyn::VolumeDataCall::ClassName(), "GetData", &AbstractVolumeManipulator::getDataCallback);
    this->outDataSlot.SetCallback(
        megamol::core::moldyn::VolumeDataCall::ClassName(), "GetExtent", &AbstractVolumeManipulator::getExtentCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->inDataSlot.SetCompatibleCall<megamol::core::moldyn::VolumeDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);

}


/*
 * datatools::AbstractVolumeManipulator::~AbstractVolumeManipulator
 */
datatools::AbstractVolumeManipulator::~AbstractVolumeManipulator(void) {}


/*
 * datatools::AbstractVolumeManipulator::create
 */
bool datatools::AbstractVolumeManipulator::create(void) {
    return true;
}


/*
 * datatools::AbstractVolumeManipulator::release
 */
void datatools::AbstractVolumeManipulator::release(void) {}


/*
 * datatools::AbstractVolumeManipulator::manipulateData
 */
bool datatools::AbstractVolumeManipulator::manipulateData(
        megamol::core::moldyn::VolumeDataCall& outData, megamol::core::moldyn::VolumeDataCall& inData) {
    outData = inData;
    inData.SetUnlocker(nullptr, false);
    return true;
}


/*
 * datatools::AbstractVolumeManipulator::manipulateExtent
 */
bool datatools::AbstractVolumeManipulator::manipulateExtent(
        megamol::core::moldyn::VolumeDataCall& outData, megamol::core::moldyn::VolumeDataCall& inData) {
    outData = inData;
    inData.SetUnlocker(nullptr, false);
    return true;
}


/*
 * datatools::AbstractVolumeManipulator::getDataCallback
 */
bool datatools::AbstractVolumeManipulator::getDataCallback(megamol::core::Call& c) {
    using megamol::core::moldyn::VolumeDataCall;

    auto* outVdc = dynamic_cast<VolumeDataCall*>(&c);
    if (outVdc == nullptr) return false;

    auto* inVdc = this->inDataSlot.CallAs<VolumeDataCall>();
    if (inVdc == nullptr) return false;

    *inVdc = *outVdc; // to get the correct request time
    if (!(*inVdc)(0)) return false;

    if (!this->manipulateData(*outVdc, *inVdc)) {
        inVdc->Unlock();
        return false;
    }

    inVdc->Unlock();

    return true;
}


/*
 * datatools::AbstractVolumeManipulator::getExtentCallback
 */
bool datatools::AbstractVolumeManipulator::getExtentCallback(megamol::core::Call& c) {
    using megamol::core::moldyn::VolumeDataCall;

    auto* outVdc = dynamic_cast<VolumeDataCall*>(&c);
    if (outVdc == nullptr) return false;
    
    auto* inVdc = this->inDataSlot.CallAs<VolumeDataCall>();
    if (inVdc == nullptr) return false;

    *inVdc = *outVdc; // to get the correct request time
    if (!(*inVdc)(1)) return false;

    if (!this->manipulateExtent(*outVdc, *inVdc)) {
        inVdc->Unlock();
        return false;
    }

    inVdc->Unlock();

    return true;
}
