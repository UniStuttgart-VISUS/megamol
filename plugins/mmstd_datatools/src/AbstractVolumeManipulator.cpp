/*
 * AbstractParticleManipulator.cpp
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmstd_datatools/AbstractParticleManipulator.h"
#include "mmstd_datatools/AbstractVolumeManipulator.h"

using namespace megamol;
using namespace megamol::stdplugin;


/*
 * datatools::AbstractVolumeManipulator::AbstractVolumeManipulator
 */
datatools::AbstractVolumeManipulator::AbstractVolumeManipulator(const char* outSlotName, const char* inSlotName)
    : megamol::core::Module()
    , outDataSlot(outSlotName, "providing access to the manipulated data")
    , inDataSlot(inSlotName, "accessing the original data") {

    this->outDataSlot.SetCallback(megamol::core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_GET_DATA),
        &AbstractVolumeManipulator::getDataCallback);
    this->outDataSlot.SetCallback(megamol::core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_GET_EXTENTS),
        &AbstractVolumeManipulator::getExtentCallback);
    this->outDataSlot.SetCallback(megamol::core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_GET_METADATA),
        &AbstractVolumeManipulator::getMetaDataCallback);
    this->outDataSlot.SetCallback(megamol::core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_START_ASYNC),
        &AbstractVolumeManipulator::startAsyncCallback);
    this->outDataSlot.SetCallback(megamol::core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_STOP_ASYNC),
        &AbstractVolumeManipulator::stopAsyncCallback);
    this->outDataSlot.SetCallback(megamol::core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_TRY_GET_DATA),
        &AbstractVolumeManipulator::tryGetDataCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->inDataSlot.SetCompatibleCall<core::misc::VolumetricDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);
}


/*
 * datatools::AbstractVolumeManipulator::~AbstractVolumeManipulator
 */
datatools::AbstractVolumeManipulator::~AbstractVolumeManipulator(void) {}


/*
 * datatools::AbstractVolumeManipulator::create
 */
bool datatools::AbstractVolumeManipulator::create(void) { return true; }


/*
 * datatools::AbstractVolumeManipulator::release
 */
void datatools::AbstractVolumeManipulator::release(void) {}


/*
 * datatools::AbstractVolumeManipulator::manipulateData
 */
bool datatools::AbstractVolumeManipulator::manipulateData(
    megamol::core::misc::VolumetricDataCall& outData, megamol::core::misc::VolumetricDataCall& inData) {
    outData = inData;
    inData.SetUnlocker(nullptr, false);
    return true;
}


/*
 * datatools::AbstractVolumeManipulator::manipulateExtent
 */
bool datatools::AbstractVolumeManipulator::manipulateExtent(
    megamol::core::misc::VolumetricDataCall& outData, megamol::core::misc::VolumetricDataCall& inData) {
    outData = inData;
    inData.SetUnlocker(nullptr, false);
    return true;
}


/*
 * datatools::AbstractVolumeManipulator::manipulateMetaData
 */
bool datatools::AbstractVolumeManipulator::manipulateMetaData(
    class core::misc::VolumetricDataCall& outData, class core::misc::VolumetricDataCall& inData) {
    outData = inData;
    inData.SetUnlocker(nullptr, false);
    return true;
}


/*
 * datatools::AbstractVolumeManipulator::getDataCallback
 */
bool datatools::AbstractVolumeManipulator::getDataCallback(megamol::core::Call& c) {
    using megamol::core::misc::VolumetricDataCall;

    auto* outVdc = dynamic_cast<VolumetricDataCall*>(&c);
    if (outVdc == nullptr) return false;

    auto* inVdc = this->inDataSlot.CallAs<VolumetricDataCall>();
    if (inVdc == nullptr) return false;

    *inVdc = *outVdc; // to get the correct request time
    if (!(*inVdc)(VolumetricDataCall::IDX_GET_DATA)){ 
      megamol::core::utility::log::Log::DefaultLog.WriteInfo("AbstractVolumeManipulator: No data available.\n");
      return false;
    }
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
    using megamol::core::misc::VolumetricDataCall;

    auto* outVdc = dynamic_cast<VolumetricDataCall*>(&c);
    if (outVdc == nullptr) return false;

    auto* inVdc = this->inDataSlot.CallAs<VolumetricDataCall>();
    if (inVdc == nullptr) return false;

    *inVdc = *outVdc; // to get the correct request time
    if (!(*inVdc)(VolumetricDataCall::IDX_GET_EXTENTS)) return false;

    if (!this->manipulateExtent(*outVdc, *inVdc)) {
        inVdc->Unlock();
        return false;
    }

    inVdc->Unlock();

    return true;
}

bool datatools::AbstractVolumeManipulator::getMetaDataCallback(megamol::core::Call& c) {
    using megamol::core::misc::VolumetricDataCall;

    auto* outVdc = dynamic_cast<VolumetricDataCall*>(&c);
    if (outVdc == nullptr) return false;

    auto* inVdc = this->inDataSlot.CallAs<VolumetricDataCall>();
    if (inVdc == nullptr) return false;

    *inVdc = *outVdc; // to get the correct request time
    if (!(*inVdc)(VolumetricDataCall::IDX_GET_METADATA)) return false;

    if (!this->manipulateMetaData(*outVdc, *inVdc)) {
        inVdc->Unlock();
        return false;
    }

    inVdc->Unlock();

    return true;
}

bool datatools::AbstractVolumeManipulator::startAsyncCallback(megamol::core::Call& c) {
    using megamol::core::misc::VolumetricDataCall;

    auto* outVdc = dynamic_cast<VolumetricDataCall*>(&c);
    if (outVdc == nullptr) return false;

    auto* inVdc = this->inDataSlot.CallAs<VolumetricDataCall>();
    if (inVdc == nullptr) return false;

    *inVdc = *outVdc; // to get the correct request time
    if (!(*inVdc)(VolumetricDataCall::IDX_START_ASYNC)) return false;

    inVdc->Unlock();

    return true;
}

bool datatools::AbstractVolumeManipulator::stopAsyncCallback(megamol::core::Call& c) {
    using megamol::core::misc::VolumetricDataCall;

    auto* outVdc = dynamic_cast<VolumetricDataCall*>(&c);
    if (outVdc == nullptr) return false;

    auto* inVdc = this->inDataSlot.CallAs<VolumetricDataCall>();
    if (inVdc == nullptr) return false;

    *inVdc = *outVdc; // to get the correct request time
    if (!(*inVdc)(VolumetricDataCall::IDX_STOP_ASYNC)) return false;

    inVdc->Unlock();

    return true;
}

bool datatools::AbstractVolumeManipulator::tryGetDataCallback(megamol::core::Call& c) {
    using megamol::core::misc::VolumetricDataCall;

    auto* outVdc = dynamic_cast<VolumetricDataCall*>(&c);
    if (outVdc == nullptr) return false;

    auto* inVdc = this->inDataSlot.CallAs<VolumetricDataCall>();
    if (inVdc == nullptr) return false;

    *inVdc = *outVdc; // to get the correct request time
    if (!(*inVdc)(VolumetricDataCall::IDX_TRY_GET_DATA)) return false;

    // TODO BUG HAZARD: if data, manipulate.

    inVdc->Unlock();

    return true;
}
